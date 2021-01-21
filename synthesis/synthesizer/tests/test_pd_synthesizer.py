from synthesis.synthesizer.tf_to_torch.torch_synthesizer import *
from commons.synthesis_program import *
import numpy as np
import re
import unittest


class TestTorchSynthesizer(unittest.TestCase):

    def test_conv2d(self):
        s_program = Program(['self.conv = tf.keras.layers.Conv2D(32,5,1)'])
        synthesizer = TorchSynthesizer(s_program, 'tf', 'torch', input=np.random.rand(100, 50, 40, 1))
        actual = synthesizer.synthesize()
        expected = Program(['self.conv = torch.nn.Conv2d(1,32,5,stride=1,padding=0,'
                            'dilation=1,groups=1,bias=True,padding_mode=\'zeros\')'])
        self.assertEqual(actual, expected)

    def test_softmax(self):
        s_program = Program(['self.softmax = tf.keras.layers.Softmax()'])
        synthesizer = TorchSynthesizer(s_program, 'tf', 'torch', input=np.random.rand(100, 50))
        actual = synthesizer.synthesize()
        expected = Program(['self.softmax = torch.nn.Softmax(dim=None)'])
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
