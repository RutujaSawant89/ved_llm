"""Environment and Configuration Sanity Tests."""

import unittest
from pathlib import Path


class TestEnvironment(unittest.TestCase):
    """Verifies that core modules and configurations are sane and loadable."""

    def test_torch_import(self):
        import torch

        x = torch.ones(2, 2)
        y = x + x
        self.assertEqual(y[0, 0].item(), 2.0)

    def test_dependencies(self):
        import matplotlib
        import numpy
        import pandas
        import seaborn
        import tiktoken
        import tqdm

        self.assertIsNotNone(numpy.__version__)
        self.assertIsNotNone(pandas.__version__)
        self.assertIsNotNone(tiktoken.__version__)
        self.assertIsNotNone(tqdm.__version__)
        self.assertIsNotNone(matplotlib.__version__)
        self.assertIsNotNone(seaborn.__version__)

    def test_config_initialization(self):
        import sys

        # Add project root to sys.path if not present
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from config import TrustLMConfig, get_default_config

        cfg = get_default_config()
        self.assertTrue(cfg.validate())
        data = cfg.to_dict()
        self.assertIn("model", data)
        self.assertIn("verification", data)
        self.assertIn("trust", data)
        self.assertIn("unlearning", data)
        self.assertIn("memory", data)


if __name__ == "__main__":
    unittest.main()
