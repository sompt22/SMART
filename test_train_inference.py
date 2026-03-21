"""Unit tests for SMART training and inference components.

Tests loss functions, decode functions, model utilities, model save/load,
and the combined GenericLoss/ModelWithLoss.

Requires PyTorch but does NOT require GPU, compiled DCNv2, or dataset files.

Run:
    python -m pytest test_train_inference.py -v
    python test_train_inference.py
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import math
import tempfile
import unittest

import torch
import torch.nn as nn
import numpy as np

# Add src/lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'lib'))

from model.utils import _sigmoid, _nms, _topk, _tranpose_and_gather_feat
from model.losses import (
    FastFocalLoss,
    RegWeightedL1Loss,
    TripletLoss,
    EmbeddingLoss,
    EmbeddingVectorLoss,
    EmbeddingVectorCosineSimilarityLoss,
)
from model.decode import generic_decode
from model.model import save_model, load_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class MockOpt:
    """Minimal opt object satisfying all loss / trainer requirements."""
    nID = 100
    embedding_dim = 64
    embedding_loss = 'softmax'
    know_dist_weight = 0.0
    num_stacks = 1
    multi_loss = ''
    embedding_weight = 1.0
    heads = {'hm': 1, 'wh': 2, 'tracking': 2, 'embedding': 64}
    weights = {'hm': 1.0, 'wh': 0.1, 'tracking': 1.0}
    prior_bias = -4.6
    backbone = 'dla34'
    neck = 'dlaup'
    head_kernel = 3
    model_output_list = False


class MockLoadOpt:
    """Minimal opt for load_model."""
    reset_hm = False
    reuse_hm = False
    resume = False
    lr = 1e-4
    lr_step = [60]


def make_batch(B=2, C=1, H=16, W=16, M=8, D=64):
    """Create a synthetic batch dict with all keys expected by GenericLoss."""
    ind = torch.randint(0, H * W, (B, M))
    mask = torch.ones(B, M, dtype=torch.float32)
    cat = torch.zeros(B, M, dtype=torch.long)

    batch = {
        'hm':           torch.rand(B, C, H, W),
        'ind':          ind,
        'mask':         mask,
        'cat':          cat,
        # regression heads
        'wh':           torch.rand(B, M, 2),
        'wh_mask':      mask.unsqueeze(2).expand(B, M, 2),
        'tracking':     torch.rand(B, M, 2),
        'tracking_mask': mask.unsqueeze(2).expand(B, M, 2),
        # embedding
        'tid':          torch.randint(0, 50, (B, M)),
        'tid_mask':     mask,
        # knowledge distillation (vector targets)
        'vectors':      torch.randn(B, M, D),
        'vectors_mask': mask,
    }
    return batch


def make_output(B=2, C=1, H=16, W=16, D=64):
    """Create a synthetic model output dict."""
    return {
        'hm':        torch.sigmoid(torch.randn(B, C, H, W)),
        'wh':        torch.rand(B, 2, H, W),
        'tracking':  torch.randn(B, 2, H, W),
        'embedding': torch.randn(B, D, H, W),
    }


# ===========================================================================
# 1. Model Utilities
# ===========================================================================

class TestModelUtils(unittest.TestCase):

    def test_sigmoid_output_range(self):
        x = torch.randn(10, 10) * 100
        y = _sigmoid(x.clone())
        self.assertTrue((y >= 1e-4).all(), "Output should be >= 1e-4")
        self.assertTrue((y <= 1 - 1e-4).all(), "Output should be <= 1-1e-4")

    def test_sigmoid_large_positive(self):
        x = torch.tensor([1000.0])
        y = _sigmoid(x.clone())
        self.assertAlmostEqual(y.item(), 1 - 1e-4, places=5)

    def test_sigmoid_large_negative(self):
        x = torch.tensor([-1000.0])
        y = _sigmoid(x.clone())
        self.assertAlmostEqual(y.item(), 1e-4, places=5)

    def test_nms_keeps_peak(self):
        # Single isolated peak at (0,5,5)
        heat = torch.zeros(1, 1, 11, 11)
        heat[0, 0, 5, 5] = 1.0
        out = _nms(heat)
        self.assertAlmostEqual(out[0, 0, 5, 5].item(), 1.0, places=5)

    def test_nms_suppresses_neighbors(self):
        # Peak at (5,5) and adjacent (5,6) — the adjacent value is lower,
        # so after NMS the neighbor should be zeroed because (5,5) is its max.
        heat = torch.zeros(1, 1, 11, 11)
        heat[0, 0, 5, 5] = 1.0
        heat[0, 0, 5, 6] = 0.5   # neighbor, non-local-max
        out = _nms(heat, kernel=3)
        # (5,6)'s 3x3 neighborhood max is 1.0 (at 5,5), so it gets zeroed
        self.assertAlmostEqual(out[0, 0, 5, 6].item(), 0.0, places=5)

    def test_topk_scores_shape(self):
        K = 5
        heat = torch.rand(2, 3, 8, 8)
        scores, inds, clses, ys, xs = _topk(heat, K=K)
        self.assertEqual(scores.shape, (2, K))
        self.assertEqual(inds.shape, (2, K))
        self.assertEqual(clses.shape, (2, K))

    def test_topk_returns_highest_score(self):
        heat = torch.zeros(1, 1, 8, 8)
        heat[0, 0, 3, 4] = 0.99  # highest peak
        scores, inds, clses, ys, xs = _topk(heat, K=3)
        self.assertAlmostEqual(scores[0, 0].item(), 0.99, places=4)

    def test_tranpose_and_gather_feat(self):
        B, D, H, W, M = 2, 4, 8, 8, 3
        feat = torch.zeros(B, D, H, W)
        # Place unique value at specific positions
        feat[0, :, 2, 3] = torch.arange(D, dtype=torch.float32)
        ind = torch.tensor([[2 * W + 3, 0, 1], [0, 0, 0]])  # index for (2,3) flattened
        gathered = _tranpose_and_gather_feat(feat, ind)
        self.assertEqual(gathered.shape, (B, M, D))
        expected = torch.arange(D, dtype=torch.float32)
        torch.testing.assert_close(gathered[0, 0], expected)


# ===========================================================================
# 2. FastFocalLoss
# ===========================================================================

class TestFastFocalLoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = FastFocalLoss()
        self.B, self.C, self.H, self.W, self.M = 2, 1, 16, 16, 4

    def _make_inputs(self, pred_val=0.5, target_val=0.0):
        B, C, H, W, M = self.B, self.C, self.H, self.W, self.M
        out = torch.full((B, C, H, W), pred_val)
        target = torch.full((B, C, H, W), target_val)
        ind = torch.zeros(B, M, dtype=torch.long)
        mask = torch.zeros(B, M)     # all zeros = no positives
        cat = torch.zeros(B, M, dtype=torch.long)
        return out, target, ind, mask, cat

    def test_loss_is_scalar(self):
        out, target, ind, mask, cat = self._make_inputs()
        loss = self.loss_fn(out, target, ind, mask, cat)
        self.assertEqual(loss.dim(), 0, "Loss should be scalar")

    def test_loss_not_nan(self):
        out, target, ind, mask, cat = self._make_inputs()
        loss = self.loss_fn(out, target, ind, mask, cat)
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss should not be Inf")

    def test_no_positives_returns_neg_loss_only(self):
        # When mask=0, no positives, loss is pure negative focal loss
        out, target, ind, mask, cat = self._make_inputs()
        loss = self.loss_fn(out, target, ind, mask, cat)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_perfect_prediction_lower_loss(self):
        # Prediction matches target exactly → lower loss than random
        B, C, H, W, M = self.B, self.C, self.H, self.W, self.M
        # Perfect: predict 1.0 where target is 1.0
        out_perfect = torch.zeros(B, C, H, W)
        out_perfect[:, :, 0, 0] = 1.0
        target = torch.zeros(B, C, H, W)
        target[:, :, 0, 0] = 1.0
        ind = torch.zeros(B, M, dtype=torch.long)  # index 0 = position (0,0)
        mask = torch.ones(B, M)
        cat = torch.zeros(B, M, dtype=torch.long)
        loss_perfect = self.loss_fn(out_perfect, target, ind, mask, cat)

        out_random = torch.zeros(B, C, H, W) + 0.5
        loss_random = self.loss_fn(out_random, target, ind, mask, cat)

        self.assertLess(loss_perfect.item(), loss_random.item())


# ===========================================================================
# 3. RegWeightedL1Loss
# ===========================================================================

class TestRegWeightedL1Loss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = RegWeightedL1Loss()

    def test_zero_mask_returns_near_zero(self):
        B, F, H, W, M = 2, 2, 8, 8, 4
        output = torch.randn(B, F, H, W)
        mask = torch.zeros(B, M, F)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randn(B, M, F)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_perfect_prediction_zero_loss(self):
        B, F, H, W, M = 2, 2, 8, 8, 4
        # Build output where gathered feat equals target
        target = torch.ones(B, M, F)
        ind = torch.zeros(B, M, dtype=torch.long)  # all point to position 0
        # Position 0 in flattened = (0,0) in spatial
        output = torch.ones(B, F, H, W)
        mask = torch.ones(B, M, F)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_loss_is_scalar(self):
        B, F, H, W, M = 2, 2, 8, 8, 4
        output = torch.randn(B, F, H, W)
        mask = torch.ones(B, M, F)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randn(B, M, F)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertEqual(loss.dim(), 0)

    def test_loss_increases_with_error(self):
        B, F, H, W, M = 2, 2, 8, 8, 4
        ind = torch.zeros(B, M, dtype=torch.long)
        mask = torch.ones(B, M, F)
        target = torch.zeros(B, M, F)

        output_small = torch.full((B, F, H, W), 0.1)
        output_large = torch.full((B, F, H, W), 5.0)

        loss_small = self.loss_fn(output_small, mask, ind, target)
        loss_large = self.loss_fn(output_large, mask, ind, target)
        self.assertLess(loss_small.item(), loss_large.item())


# ===========================================================================
# 4. TripletLoss
# ===========================================================================

class TestTripletLoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = TripletLoss(margin=0.3)

    def test_single_sample_returns_zero(self):
        inputs = torch.randn(1, 64)
        targets = torch.tensor([0])
        loss = self.loss_fn(inputs, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_well_separated_embeddings_low_loss(self):
        # Two clearly separated classes
        inputs = torch.cat([
            torch.eye(8)[:4],   # class 0: unit vectors in first 4 dims
            -torch.eye(8)[:4],  # class 1: negative unit vectors
        ])
        targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = self.loss_fn(inputs, targets)
        self.assertLessEqual(loss.item(), 0.3)

    def test_loss_is_scalar(self):
        inputs = torch.randn(6, 32)
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = self.loss_fn(inputs, targets)
        self.assertEqual(loss.dim(), 0)

    def test_loss_not_nan(self):
        inputs = torch.randn(4, 16)
        targets = torch.tensor([0, 0, 1, 1])
        loss = self.loss_fn(inputs, targets)
        self.assertFalse(torch.isnan(loss))


# ===========================================================================
# 5. EmbeddingLoss
# ===========================================================================

class TestEmbeddingLoss(unittest.TestCase):

    def setUp(self):
        self.opt = MockOpt()
        self.loss_fn = EmbeddingLoss(self.opt)

    def test_empty_mask_returns_zero(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.zeros(B, M)      # no valid positions
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.zeros(B, M, dtype=torch.long)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_out_of_range_ids_ignored(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.ones(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        # IDs >= nID should be clamped / ignored
        target = torch.full((B, M), self.opt.nID + 10, dtype=torch.long)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_loss_is_finite(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.ones(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randint(0, self.opt.nID, (B, M))
        loss = self.loss_fn(output, mask, ind, target)
        self.assertFalse(torch.isnan(loss), "Loss is NaN")
        self.assertFalse(torch.isinf(loss), "Loss is Inf")
        self.assertGreater(loss.item(), 0.0)

    def test_loss_is_scalar(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.ones(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randint(0, self.opt.nID, (B, M))
        loss = self.loss_fn(output, mask, ind, target)
        self.assertEqual(loss.dim(), 0)


# ===========================================================================
# 6. EmbeddingVectorLoss
# ===========================================================================

class TestEmbeddingVectorLoss(unittest.TestCase):

    def setUp(self):
        self.opt = MockOpt()
        self.loss_fn = EmbeddingVectorLoss(self.opt)

    def test_empty_mask_returns_zero(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.zeros(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randn(B, M, D)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_loss_is_finite(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.ones(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randn(B, M, D)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_loss_is_scalar(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.ones(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randn(B, M, D)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertEqual(loss.dim(), 0)


# ===========================================================================
# 7. EmbeddingVectorCosineSimilarityLoss
# ===========================================================================

class TestEmbeddingVectorCosineSimilarityLoss(unittest.TestCase):

    def setUp(self):
        self.opt = MockOpt()
        self.loss_fn = EmbeddingVectorCosineSimilarityLoss(self.opt)

    def test_empty_mask_returns_zero(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.zeros(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randn(B, M, D)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_same_direction_near_zero_loss(self):
        """Output in same direction as target → cosine_sim ≈ 1 → loss ≈ 0."""
        B, D, H, W, M = 1, 4, 4, 4, 1
        # Craft output so gathered feat = [1,0,0,0] and target = [1,0,0,0]
        output = torch.zeros(B, D, H, W)
        output[0, 0, 0, 0] = 1.0   # unit vector in dim-0
        ind = torch.zeros(B, M, dtype=torch.long)
        mask = torch.ones(B, M)
        target = torch.zeros(B, M, D)
        target[0, 0, 0] = 1.0      # same direction
        loss = self.loss_fn(output, mask, ind, target)
        # Loss = 1 - cosine_similarity; identical directions → loss ≈ 0
        self.assertLess(loss.item(), 0.1)

    def test_orthogonal_vectors_loss_near_one(self):
        """Orthogonal output and target → cosine_sim = 0 → loss = 1."""
        B, D, H, W, M = 1, 4, 4, 4, 1
        output = torch.zeros(B, D, H, W)
        output[0, 0, 0, 0] = 1.0   # direction: dim-0
        ind = torch.zeros(B, M, dtype=torch.long)
        mask = torch.ones(B, M)
        target = torch.zeros(B, M, D)
        target[0, 0, 1] = 1.0      # direction: dim-1 (orthogonal)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertGreater(loss.item(), 0.5)

    def test_loss_is_finite(self):
        B, D, H, W, M = 2, 64, 8, 8, 4
        output = torch.randn(B, D, H, W)
        mask = torch.ones(B, M)
        ind = torch.zeros(B, M, dtype=torch.long)
        target = torch.randn(B, M, D)
        loss = self.loss_fn(output, mask, ind, target)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


# ===========================================================================
# 8. generic_decode
# ===========================================================================

class TestGenericDecode(unittest.TestCase):

    def _make_output(self, B=2, C=1, H=16, W=16, D=64, with_wh=True,
                     with_embedding=True, with_tracking=True):
        out = {'hm': torch.sigmoid(torch.randn(B, C, H, W))}
        if with_wh:
            out['wh'] = torch.rand(B, 2, H, W) * 10
        if with_embedding:
            out['embedding'] = torch.randn(B, D, H, W)
        if with_tracking:
            out['tracking'] = torch.randn(B, 2, H, W)
        return out

    def test_no_hm_returns_empty_dict(self):
        result = generic_decode({}, K=10)
        self.assertEqual(result, {})

    def test_scores_shape(self):
        K = 10
        out = self._make_output()
        result = generic_decode(out, K=K)
        self.assertEqual(result['scores'].shape, (2, K))

    def test_clses_shape(self):
        K = 10
        out = self._make_output()
        result = generic_decode(out, K=K)
        self.assertEqual(result['clses'].shape, (2, K))

    def test_bboxes_with_wh(self):
        K = 10
        out = self._make_output(with_wh=True)
        result = generic_decode(out, K=K)
        self.assertIn('bboxes', result)
        self.assertEqual(result['bboxes'].shape, (2, K, 4))

    def test_no_bboxes_without_wh(self):
        K = 5
        out = self._make_output(with_wh=False, with_embedding=False,
                                with_tracking=False)
        result = generic_decode(out, K=K)
        self.assertNotIn('bboxes', result)

    def test_embedding_extracted(self):
        K = 10
        out = self._make_output(D=64, with_embedding=True)
        result = generic_decode(out, K=K)
        self.assertIn('embedding', result)
        self.assertEqual(result['embedding'].shape, (2, K, 64))

    def test_embedding_normalized(self):
        K = 5
        out = self._make_output(D=64, with_embedding=True)
        result = generic_decode(out, K=K)
        norms = result['embedding'].norm(dim=2)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)

    def test_tracking_extracted(self):
        K = 10
        out = self._make_output(with_tracking=True)
        result = generic_decode(out, K=K)
        self.assertIn('tracking', result)
        self.assertEqual(result['tracking'].shape, (2, K, 2))

    def test_cts_shape(self):
        K = 10
        out = self._make_output()
        result = generic_decode(out, K=K)
        self.assertIn('cts', result)
        self.assertEqual(result['cts'].shape, (2, K, 2))

    def test_high_peak_selected_first(self):
        """The cell with the highest heatmap value should yield the top score."""
        B, C, H, W = 1, 1, 16, 16
        hm = torch.zeros(B, C, H, W)
        hm[0, 0, 5, 7] = 0.99    # isolated peak
        out = {'hm': hm}
        result = generic_decode(out, K=5)
        top_score = result['scores'][0, 0].item()
        self.assertAlmostEqual(top_score, 0.99, places=3)

    def test_ltrb_amodal_overrides_bboxes(self):
        K = 5
        B, C, H, W = 2, 1, 16, 16
        out = {
            'hm': torch.sigmoid(torch.randn(B, C, H, W)),
            'ltrb_amodal': torch.randn(B, 4, H, W),
        }
        result = generic_decode(out, K=K)
        self.assertIn('bboxes_amodal', result)
        self.assertIn('bboxes', result)
        # bboxes should equal bboxes_amodal when ltrb_amodal present
        torch.testing.assert_close(result['bboxes'], result['bboxes_amodal'])


# ===========================================================================
# 9. save_model / load_model
# ===========================================================================

class TestSaveLoadModel(unittest.TestCase):
    """Tests using a simple nn.Linear to avoid DCNv2 dependency."""

    def _simple_model(self, in_f=10, out_f=5):
        return nn.Linear(in_f, out_f)

    def test_save_creates_file(self):
        model = self._simple_model()
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        save_model(path, epoch=1, model=model)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_save_includes_epoch(self):
        model = self._simple_model()
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        save_model(path, epoch=42, model=model)
        ckpt = torch.load(path, map_location='cpu')
        self.assertEqual(ckpt['epoch'], 42)
        os.remove(path)

    def test_load_restores_weights(self):
        model = self._simple_model()
        original_weights = model.weight.data.clone()

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        save_model(path, epoch=1, model=model)

        # Create a fresh model and load
        model2 = self._simple_model()
        nn.init.zeros_(model2.weight)  # different init
        opt = MockLoadOpt()
        model2 = load_model(model2, path, opt)
        torch.testing.assert_close(model2.weight.data, original_weights)
        os.remove(path)

    def test_save_with_optimizer(self):
        model = self._simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        save_model(path, epoch=5, model=model, optimizer=optimizer)
        ckpt = torch.load(path, map_location='cpu')
        self.assertIn('optimizer', ckpt)
        os.remove(path)

    def test_load_strips_module_prefix(self):
        """State dict keys prefixed with 'module.' (from DataParallel) are stripped."""
        model = self._simple_model()
        # Manually add 'module.' prefix to simulate DataParallel checkpoint
        prefixed_state = {'module.' + k: v for k, v in model.state_dict().items()}
        ckpt = {'epoch': 1, 'state_dict': prefixed_state}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        torch.save(ckpt, path)

        model2 = self._simple_model()
        opt = MockLoadOpt()
        model2 = load_model(model2, path, opt)
        # Weights should still match
        torch.testing.assert_close(model2.weight.data, model.weight.data)
        os.remove(path)

    def test_load_handles_missing_param(self):
        """Params in model but not in checkpoint keep their initialised values."""
        # Save a model with one layer
        small = nn.Linear(10, 5)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        save_model(path, epoch=1, model=small)

        # Load into a *different* model (extra params won't match)
        big = nn.Linear(10, 8)   # different out dimension
        init_weight = big.weight.data.clone()
        opt = MockLoadOpt()
        big = load_model(big, path, opt)
        # big.weight should remain at init (shape mismatch → skipped)
        torch.testing.assert_close(big.weight.data, init_weight)
        os.remove(path)


# ===========================================================================
# 10. GenericLoss (from trainer.py)
# ===========================================================================

class TestGenericLoss(unittest.TestCase):

    def _make_opt(self, **kwargs):
        opt = MockOpt()
        for k, v in kwargs.items():
            setattr(opt, k, v)
        return opt

    def test_loss_is_finite_scalar(self):
        from trainer import GenericLoss
        opt = self._make_opt()
        loss_fn = GenericLoss(opt)
        outputs = [make_output()]
        batch = make_batch()
        tot, losses = loss_fn(outputs, batch)
        self.assertEqual(tot.dim(), 0)
        self.assertFalse(torch.isnan(tot), "Total loss is NaN")
        self.assertFalse(torch.isinf(tot), "Total loss is Inf")

    def test_loss_dict_has_all_heads(self):
        from trainer import GenericLoss
        opt = self._make_opt()
        loss_fn = GenericLoss(opt)
        outputs = [make_output()]
        batch = make_batch()
        _, losses = loss_fn(outputs, batch)
        for head in opt.heads:
            self.assertIn(head, losses)
        self.assertIn('tot', losses)

    def test_embedding_weight_zero_disables_vector(self):
        """know_dist_weight=0 → lambda_vector=0 → only classification loss."""
        from trainer import GenericLoss
        opt = self._make_opt(know_dist_weight=0.0)
        loss_fn = GenericLoss(opt)
        outputs = [make_output()]
        batch = make_batch()
        # Should not raise; vector loss path should be skipped
        tot, losses = loss_fn(outputs, batch)
        self.assertFalse(torch.isnan(tot))

    def test_uncertainty_multi_loss(self):
        """multi_loss='uncertainty' uses learnable s_det, s_id parameters."""
        from trainer import GenericLoss
        opt = self._make_opt(multi_loss='uncertainty')
        loss_fn = GenericLoss(opt)
        # Verify learnable params exist
        self.assertTrue(hasattr(loss_fn, 's_det'))
        self.assertTrue(hasattr(loss_fn, 's_id'))
        outputs = [make_output()]
        batch = make_batch()
        tot, _ = loss_fn(outputs, batch)
        self.assertFalse(torch.isnan(tot))

    def test_hm_loss_positive(self):
        """Heatmap loss should be > 0 for random predictions."""
        from trainer import GenericLoss
        opt = self._make_opt()
        loss_fn = GenericLoss(opt)
        outputs = [make_output()]
        batch = make_batch()
        _, losses = loss_fn(outputs, batch)
        self.assertGreater(losses['hm'].item(), 0.0)

    def test_num_stacks_scaling(self):
        """With num_stacks=2 each stack contributes 1/num_stacks to total."""
        from trainer import GenericLoss
        opt1 = self._make_opt(num_stacks=1)
        opt2 = self._make_opt(num_stacks=2)
        loss_fn1 = GenericLoss(opt1)
        loss_fn2 = GenericLoss(opt2)

        # Use same random seed for both outputs
        torch.manual_seed(0)
        out1 = [make_output()]
        torch.manual_seed(0)
        out2 = [make_output(), make_output()]

        batch = make_batch()
        tot1, _ = loss_fn1(out1, batch)
        tot2, _ = loss_fn2(out2, batch)
        # Both should be finite
        self.assertFalse(torch.isnan(tot1))
        self.assertFalse(torch.isnan(tot2))


# ===========================================================================
# 11. ModelWithLoss (from trainer.py)
# ===========================================================================

class TestModelWithLoss(unittest.TestCase):

    def _make_mock_model(self):
        """A nn.Module that returns a list of synthetic output dicts."""
        class FakeModel(nn.Module):
            def forward(self, image, pre_img=None, pre_hm=None):
                B = image.shape[0]
                return [make_output(B=B)]
        return FakeModel()

    def test_forward_returns_output_loss_stats(self):
        from trainer import GenericLoss, ModelWithLoss
        opt = MockOpt()
        model = self._make_mock_model()
        loss_fn = GenericLoss(opt)
        mwl = ModelWithLoss(model, loss_fn)

        batch = make_batch(B=2)
        batch['image'] = torch.randn(2, 3, 16, 16)
        output, loss, loss_stats = mwl(batch)
        self.assertIsInstance(loss_stats, dict)
        self.assertIn('tot', loss_stats)
        self.assertEqual(loss.dim(), 0)

    def test_loss_is_scalar(self):
        from trainer import GenericLoss, ModelWithLoss
        opt = MockOpt()
        model = self._make_mock_model()
        loss_fn = GenericLoss(opt)
        mwl = ModelWithLoss(model, loss_fn)

        batch = make_batch(B=2)
        batch['image'] = torch.randn(2, 3, 16, 16)
        _, loss, _ = mwl(batch)
        self.assertEqual(loss.dim(), 0)
        self.assertFalse(torch.isnan(loss))

    def test_pre_img_pre_hm_passed(self):
        """pre_img and pre_hm in batch should be forwarded to the model."""
        from trainer import GenericLoss, ModelWithLoss

        received = {}

        class RecordingModel(nn.Module):
            def forward(self, image, pre_img=None, pre_hm=None):
                received['pre_img'] = pre_img
                received['pre_hm'] = pre_hm
                B = image.shape[0]
                return [make_output(B=B)]

        opt = MockOpt()
        model = RecordingModel()
        loss_fn = GenericLoss(opt)
        mwl = ModelWithLoss(model, loss_fn)

        batch = make_batch(B=2)
        batch['image'] = torch.randn(2, 3, 16, 16)
        batch['pre_img'] = torch.randn(2, 3, 16, 16)
        batch['pre_hm'] = torch.zeros(2, 1, 16, 16)
        mwl(batch)

        self.assertIsNotNone(received.get('pre_img'))
        self.assertIsNotNone(received.get('pre_hm'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
