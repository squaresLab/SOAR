import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest

from numpy.testing import assert_allclose
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model


class TFConv(Model):
  def __init__(self, constant_init):
    super(TFConv, self).__init__()
    self.conv1 = Conv2D(32, 3, strides=2, use_bias=True,
                        kernel_initializer=tf.initializers.Constant(constant_init),
                        bias_initializer = tf.initializers.Constant(constant_init))

  def call(self, x):
    x = self.conv1(x)
    return x


class TorchConv(nn.Module):
  def __init__(self, constant_init):
    super(TorchConv, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride=2, bias=True)
    torch.nn.init.constant_(self.conv1.weight.data, constant_init)
    torch.nn.init.constant_(self.conv1.bias.data, constant_init)


  def forward(self, x):
    x = x.permute(0, 3, 1, 2)
    x = self.conv1(x)  # with parameters, random init
    x = x.permute(0, 2, 3, 1)
    return x


class ConsistenceTests(unittest.TestCase):

  def test_conv(self):
    # input for the network
    const_init = 0.2
    # dataset-size, 50-width, 40-height, 1-channels
    imgs = np.random.rand(100, 50, 40, 1)

    # get the result from Tensorflow
    tf_conv_model = TFConv(const_init)
    tf_input = tf.convert_to_tensor(imgs)
    tf_x = tf_conv_model(tf_input)
    tf_result = tf_x.numpy()

    # get the result from Torch
    torch_conv_model = TorchConv(const_init)
    torch_input = torch.Tensor(imgs)
    torch_x = torch_conv_model(torch_input)
    torch_result = torch_x.detach().numpy()

    # check if the results are consistent
    assert_allclose(torch_result, tf_result, atol=1e-5)
