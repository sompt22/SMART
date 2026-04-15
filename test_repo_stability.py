"""Regression tests for researcher-facing workflow stability.

These checks avoid heavy runtime dependencies and focus on source-level
guarantees for the documented CLI and helper scripts.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'lib'))

from opts import opts


class TestCliParsing(unittest.TestCase):
    def test_display_flag_parses(self):
        opt = opts().parse(['tracking,embedding', '--display', '--gpus', '-1'])
        self.assertTrue(opt.display)
        self.assertEqual(opt.gpus, [-1])


class TestReadmeDocs(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
            self.readme = f.read()

    def test_readme_references_existing_dataset_scripts(self):
        self.assertIn('bash src/tools/mot17/get_mot_17.sh', self.readme)
        self.assertIn('bash src/tools/mot20/get_mot_20.sh', self.readme)
        self.assertIn('python3 src/tools/convert_crowdhuman_to_coco.py', self.readme)

    def test_readme_does_not_reference_missing_or_removed_cli(self):
        self.assertNotIn('--teacher_model', self.readme)
        self.assertNotIn('convert_kitti_to_coco.py', self.readme)
        self.assertNotIn('convert_nuscenes_to_coco.py', self.readme)


class TestHelperScripts(unittest.TestCase):
    def _read(self, rel_path):
        path = os.path.join(os.path.dirname(__file__), rel_path)
        with open(path) as f:
            return f.read()

    def test_train_script_is_portable(self):
        content = self._read('experiments/train.sh')
        self.assertIn('python3 main.py', content)
        self.assertIn('--same_aug_pre', content)
        self.assertNotIn('nvidia-smi', content)

    def test_inference_defaults_to_headless(self):
        content = self._read('src/demo.py')
        self.assertIn('if opt.display:', content)
        self.assertNotIn('opt.debug = max(opt.debug, 1)', content)

    def test_opts_init_uses_passed_args_and_tracking_default(self):
        content = self._read('src/lib/opts.py')
        self.assertIn('opt = self.parse(args)', content)
        self.assertIn("'tracking,embedding': 'mot17'", content)

    def test_legacy_mot17_converter_is_deprecated_wrapper(self):
        content = self._read('src/tools/mot17/convert_mot_to_coco.py')
        self.assertIn('Deprecated entrypoint', content)
        self.assertIn("runpy.run_path", content)

    def test_docker_uses_explicit_data_dir_override(self):
        compose = self._read('docker-compose.yml')
        dockerfile = self._read('Dockerfile')
        self.assertIn('SMART_DATA_DIR=/data', compose)
        self.assertIn('SMART_DATA_DIR=/data', dockerfile)

    def test_opts_supports_data_dir_override(self):
        content = self._read('src/lib/opts.py')
        self.assertIn("'SMART_DATA_DIR'", content)
        self.assertIn("os.environ.get(", content)

    def test_divo_and_sompt22_converters_are_repo_relative(self):
        for rel_path in [
            'src/tools/divo/convert_mot_to_coco.py',
            'src/tools/sompt22/convert_mot_to_coco.py',
        ]:
            content = self._read(rel_path)
            self.assertIn('Path(__file__).resolve()', content)
            self.assertNotIn('/media/hdd4tb/', content)

    def test_crowdhuman_paths_are_flexible(self):
        loader = self._read('src/lib/dataset/datasets/crowdhuman.py')
        converter = self._read('src/tools/convert_crowdhuman_to_coco.py')
        self.assertIn('def _resolve_img_dir', loader)
        self.assertIn('CrowdHuman_{}', loader)
        self.assertIn('resolve_image_root', converter)


if __name__ == '__main__':
    unittest.main()
