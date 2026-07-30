import unittest

import torch
from easydict import EasyDict

from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.depth_auxiliary import depth_auxiliary_loss


class DepthAuxiliaryTest(unittest.TestCase):
    def _config(self, enabled):
        return EasyDict({
            "num_classes": 2,
            "input_channel": 3,
            "embedding_size": 128,
            "input_size": [80, 80],
            "depth_aux_enabled": enabled,
            "depth_target_size": (40, 40),
        })

    def test_disabled_keeps_two_outputs(self):
        model = MultiFTNet(conf=self._config(False), num_classes=2, img_channel=3, conv6_kernel=(5, 5))
        model.train()
        outputs = model(torch.randn(2, 3, 80, 80))
        self.assertEqual(len(outputs), 2)

    def test_enabled_returns_depth_map(self):
        model = MultiFTNet(conf=self._config(True), num_classes=2, img_channel=3, conv6_kernel=(5, 5))
        model.train()
        cls, ft, depth = model(torch.randn(2, 3, 80, 80))
        self.assertEqual(tuple(cls.shape), (2, 2))
        self.assertEqual(tuple(ft.shape), (2, 1, 10, 10))
        self.assertEqual(tuple(depth.shape), (2, 1, 40, 40))

    def test_depth_loss_backpropagates(self):
        prediction = torch.sigmoid(torch.randn(2, 1, 40, 40, requires_grad=True))
        target = torch.rand_like(prediction)
        loss, _, _ = depth_auxiliary_loss(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
