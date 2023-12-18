import torch
import torch.nn.functional as F
import os
import unittest
from torch.testing._internal.zkp_test_lists import CustomTestCase


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(CustomTestCase))
    runner = unittest.TextTestRunner()
    runner.run(suite)
