"""Tests for model, losses, dataset, and training improvements.
Source-code verification tests that don't require torch installation.
"""
import sys
import os
import unittest

# Add src/lib to path for import checks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'lib'))


class TestModelLoad(unittest.TestCase):
    """Test model loading fixes."""

    def test_load_model_reuse_hm_comparison(self):
        """The reuse_hm comparison should compare with model_state_dict, not self."""
        model_path = os.path.join(os.path.dirname(__file__),
                                   'src', 'lib', 'model', 'model.py')
        with open(model_path) as f:
            content = f.read()

        # Should NOT contain self-comparison
        self.assertNotIn(
            'state_dict[k].shape[0] < state_dict[k].shape[0]', content)
        # Should contain correct comparison
        self.assertIn(
            'state_dict[k].shape[0] < model_state_dict[k].shape[0]', content)


class TestLossesSource(unittest.TestCase):
    """Test loss function source code fixes."""

    def _read_losses(self):
        path = os.path.join(os.path.dirname(__file__),
                            'src', 'lib', 'model', 'losses.py')
        with open(path) as f:
            return f.read()

    def test_no_elementwise_mean(self):
        """No deprecated 'elementwise_mean' reduction should remain."""
        content = self._read_losses()
        self.assertNotIn("reduction='elementwise_mean'", content)

    def test_addmm_uses_keyword_args(self):
        """addmm_ should use keyword args (beta=, alpha=)."""
        content = self._read_losses()
        self.assertNotIn('addmm_(1, -2,', content)
        self.assertIn('addmm_(inputs, inputs.t(), beta=1, alpha=-2)', content)

    def test_no_slow_neg_loss(self):
        """_slow_neg_loss should be removed."""
        content = self._read_losses()
        self.assertNotIn('def _slow_neg_loss', content)

    def test_no_not_faster_neg_loss(self):
        """_not_faster_neg_loss should be removed."""
        content = self._read_losses()
        self.assertNotIn('def _not_faster_neg_loss', content)

    def test_no_slow_reg_loss(self):
        """_slow_reg_loss should be removed."""
        content = self._read_losses()
        self.assertNotIn('def _slow_reg_loss', content)

    def test_no_neg_loss(self):
        """_neg_loss should be removed (only used by removed FocalLoss)."""
        content = self._read_losses()
        self.assertNotIn('def _neg_loss', content)

    def test_no_reg_loss(self):
        """_reg_loss should be removed."""
        content = self._read_losses()
        self.assertNotIn('def _reg_loss', content)

    def test_no_focal_loss_class(self):
        """FocalLoss (non-Fast) should be removed."""
        content = self._read_losses()
        self.assertNotIn('class FocalLoss', content)

    def test_no_reg_loss_class(self):
        """RegLoss class should be removed."""
        content = self._read_losses()
        self.assertNotIn('class RegLoss(', content)

    def test_no_regl1_loss_class(self):
        """RegL1Loss should be removed."""
        content = self._read_losses()
        self.assertNotIn('class RegL1Loss', content)

    def test_no_norml1_loss_class(self):
        """NormRegL1Loss should be removed."""
        content = self._read_losses()
        self.assertNotIn('class NormRegL1Loss', content)

    def test_no_l1_loss_class(self):
        """L1Loss should be removed."""
        content = self._read_losses()
        self.assertNotIn('class L1Loss', content)

    def test_fast_focal_loss_exists(self):
        """FastFocalLoss should still exist."""
        content = self._read_losses()
        self.assertIn('class FastFocalLoss', content)

    def test_reg_weighted_l1_loss_exists(self):
        """RegWeightedL1Loss should still exist."""
        content = self._read_losses()
        self.assertIn('class RegWeightedL1Loss', content)

    def test_embedding_loss_exists(self):
        """EmbeddingLoss should still exist."""
        content = self._read_losses()
        self.assertIn('class EmbeddingLoss', content)

    def test_triplet_loss_exists(self):
        """TripletLoss should still exist."""
        content = self._read_losses()
        self.assertIn('class TripletLoss', content)

    def test_embedding_vector_loss_uses_normalized(self):
        """EmbeddingVectorLoss should use vector_head_normalized for MSE."""
        content = self._read_losses()
        # Find the EmbeddingVectorLoss class
        start = content.find('class EmbeddingVectorLoss')
        end = content.find('class ', start + 1)
        evl_code = content[start:end]
        # Should use normalized for loss
        self.assertIn('mse_loss(vector_head_normalized', evl_code)
        # Should NOT use non-normalized for loss
        self.assertNotIn('mse_loss(vector_head_masked', evl_code)

    def test_no_unused_imports(self):
        """Unused imports (_nms, _topk, draw_umich_gaussian) should be removed."""
        content = self._read_losses()
        self.assertNotIn('_nms', content)
        self.assertNotIn('_topk', content)
        self.assertNotIn('draw_umich_gaussian', content)

    def test_no_debug_prints_in_vector_losses(self):
        """Debug print blocks should be cleaned from vector loss classes."""
        content = self._read_losses()
        # EmbeddingVectorLoss and CosineSimilarityLoss should not have debug prints
        start = content.find('class EmbeddingVectorLoss')
        end = content.find('class ', start + 1)
        evl_code = content[start:end]
        self.assertNotIn('print("vector head shape', evl_code)

        start = content.find('class EmbeddingVectorCosineSimilarityLoss')
        cosine_code = content[start:]
        self.assertNotIn('print("vector head shape', cosine_code)


class TestTrainerFixes(unittest.TestCase):
    """Test trainer improvements."""

    def _read_trainer(self):
        path = os.path.join(os.path.dirname(__file__),
                            'src', 'lib', 'trainer.py')
        with open(path) as f:
            return f.read()

    def test_model_with_loss_renamed(self):
        """Class should be ModelWithLoss not ModleWithLoss."""
        content = self._read_trainer()
        self.assertIn('class ModelWithLoss', content)
        self.assertNotIn('class ModleWithLoss', content)

    def test_gradient_clipping_present(self):
        """Training loop should include gradient clipping."""
        content = self._read_trainer()
        self.assertIn('clip_grad_norm_', content)

    def test_zero_grad_set_to_none(self):
        """Training loop should use set_to_none=True."""
        content = self._read_trainer()
        self.assertIn('set_to_none=True', content)

    def test_model_with_loss_reference_updated(self):
        """All references to ModleWithLoss should be updated."""
        content = self._read_trainer()
        self.assertNotIn('ModleWithLoss', content)


class TestDatasetFixes(unittest.TestCase):
    """Test dataset fixes."""

    def _read_dataset(self):
        path = os.path.join(os.path.dirname(__file__),
                            'src', 'lib', 'dataset', 'generic_dataset.py')
        with open(path) as f:
            return f.read()

    def test_return_hm_typo_fixed(self):
        """reutrn_hm typo should be fixed to return_hm."""
        content = self._read_dataset()
        self.assertNotIn('reutrn_hm', content)
        self.assertIn('return_hm', content)

    def test_isinstance_used(self):
        """Should use isinstance() instead of type() ==."""
        content = self._read_dataset()
        self.assertNotIn("type(s) == float", content)
        self.assertIn("isinstance(s, (int, float))", content)

    def test_embedding_comparison_fixed(self):
        """Should not use 'ann[embedding] in embedding_vectors' for numpy arrays."""
        content = self._read_dataset()
        self.assertNotIn("ann['embedding'] in embedding_vectors", content)

    def test_aug_param_simplified(self):
        """After scalar-to-list conversion, both branches should use s[0]/s[1]."""
        content = self._read_dataset()
        # After isinstance check converts scalar to list, code should use s[0]
        aug_section = content[content.find('def _get_aug_param'):]
        aug_section = aug_section[:aug_section.find('\n  def ')]
        self.assertIn('s[0]', aug_section)
        self.assertIn('s[1]', aug_section)


class TestMainPy(unittest.TestCase):
    """Test main.py cleanup."""

    def _read_main(self):
        path = os.path.join(os.path.dirname(__file__), 'src', 'main.py')
        with open(path) as f:
            return f.read()

    def test_no_dead_optimizer_functions(self):
        """get_optimizer__ and get_optimizer_separate should be removed."""
        content = self._read_main()
        self.assertNotIn('def get_optimizer__', content)
        self.assertNotIn('def get_optimizer_separate', content)
        # But get_optimizer should still exist
        self.assertIn('def get_optimizer', content)


class TestHeadConstruction(unittest.TestCase):
    """Test that head construction uses loop-based approach."""

    def _read_network(self, filename):
        path = os.path.join(os.path.dirname(__file__),
                            'src', 'lib', 'model', 'networks', filename)
        with open(path) as f:
            return f.read()

    def test_generic_network_no_elif_chain(self):
        """generic_network.py should not have if/elif chain for convs."""
        content = self._read_network('generic_network.py')
        # The old code had "if len(convs) == 1:" etc.
        self.assertNotIn('if len(convs) == 1', content)
        self.assertNotIn('elif len(convs) == 2', content)
        # Should use loop-based approach
        self.assertIn('for c in convs:', content)

    def test_base_model_no_elif_chain(self):
        """base_model.py should not have if/elif chain for convs."""
        content = self._read_network('base_model.py')
        self.assertNotIn('if len(convs) == 1', content)
        self.assertNotIn('elif len(convs) == 2', content)
        self.assertIn('for c in convs:', content)

    def test_generic_network_uses_sequential_star(self):
        """Should use nn.Sequential(*layers) pattern."""
        content = self._read_network('generic_network.py')
        self.assertIn('nn.Sequential(*layers)', content)

    def test_base_model_uses_sequential_star(self):
        """Should use nn.Sequential(*layers) pattern."""
        content = self._read_network('base_model.py')
        self.assertIn('nn.Sequential(*layers)', content)


if __name__ == '__main__':
    unittest.main()
