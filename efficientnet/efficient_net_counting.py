# -*- coding: utf-8 -*-
"""Efficient Net: Counting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/google-research/google-research/blob/master/micronet_challenge/EfficientNetCounting.ipynb

##### Copyright 2019 MicroNet Challenge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
"""

# Copyright 2019 MicroNet Challenge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License atte
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""# Counting the Parameters and Operations for EfficientNet
Here is the plan
1. Create an instance of tf.keras.Model implementation of [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) using `create_model()`. 
2. Extract the operations manually from the compiled model into the framework-agnostic API defined in `micronet_challenge.counting` using ``read_model()`
3. Using the operations in the given list, we print total parameter count using `micronet_challenge.counting.MicroNetCounter()` class.

Let's start with creating the model and running an input of ones through.
"""

# Commented out IPython magic to ensure Python compatibility.
# # Download the official EfficientNet implementation and add an init file to
# # the EfficientNet module s.t. we can use the model builders for our counting.
# %%bash 
# test -d tpu || git clone https://github.com/tensorflow/tpu tpu && mv tpu/models/official/efficientnet/* ./ 
# test -d gresearch || git clone https://github.com/google-research/google-research gresearch && mv gresearch/micronet_challenge ./

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import counting
import efficientnet_builder
import efficientnet_model

"""## 1. Creating the Model"""

"""Creates and read the operations of EfficientNet instances.
"""
DEFAULT_INPUT_SIZES = {
    # (width_coefficient, depth_coefficient, resolution)
    'efficientnet-b0': 224,
    'efficientnet-b1': 240,
    'efficientnet-b2': 260,
    'efficientnet-b3': 300,
    'efficientnet-b4': 380,
    'efficientnet-b5': 456,
    'efficientnet-b6': 528,
    'efficientnet-b7': 600}

def create_model(model_name, input_shape=None):
  """Creates and reads operations from the given model.

  Args:
    model_name: str, one of the DEFAULT_INPUT_SIZES.keys()
    input_shape: str or None, if None will be read from the dictionary.

  Returns:
    list, of operations.
  """
  if input_shape is None:
    input_size = DEFAULT_INPUT_SIZES[model_name]
    input_shape = (1, input_size, input_size, 3)
  blocks_args, global_params = efficientnet_builder.get_model_params(
      model_name, None)

  print('global_params= %s' % str(global_params))
  print('blocks_args= %s' % str('\n'.join(map(str, blocks_args))))
  tf.reset_default_graph()
  with tf.variable_scope(model_name):
    model = efficientnet_model.Model(blocks_args, global_params)
  # This will initialize the variables.
  _ = model(tf.ones((input_shape)))
  return model, input_shape

model_name = 'efficientnet-b3'
model, input_shape = create_model(model_name)

"""## 2. Extracting Operations
- We assume 'same' padding with square images/conv kernels.
- batchnorm scales are not counted since they can be merged. Bias added for each batch norm applied on a layer's output.
- `f_activation` can be changed to one of the followin `relu` or `swish`.
"""

#@title Reading Utils
# assumes everything square
# returns number of pixels for which a convolution is calculated


def read_block(block, input_size, f_activation='swish'):
  """Reads the operations on a single EfficientNet block.

  Args:
    block: efficientnet_model.MBConvBlock,
    input_shape: int, square image assumed.
    f_activation: str or None, one of 'relu', 'swish', None.

  Returns:
    list, of operations.
  """

  conv_counter_class = {
      efficientnet_model.MBConvBlock: counting.Conv2D,
      efficientnet_model.MBConvBlockBinary: counting.Conv2DBinary,
  }[type(block)]

  ops = []
  # 1
  l_name = '_expand_conv'
  if hasattr(block, l_name):
    layer = getattr(block, l_name)
    layer_temp = conv_counter_class(
        input_size, layer.kernel.shape.as_list(), layer.strides, layer.padding,
        True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
  # 2
  l_name = '_depthwise_conv'
  layer = getattr(block, l_name)
  layer_temp = counting.DepthWiseConv2D(
      input_size, layer.weights[0].shape.as_list(), layer.strides,
      layer.padding, True, f_activation)  # Use bias true since batch_norm
  ops.append((l_name, layer_temp))
  # Input size might have changed.
  input_size = counting.get_conv_output_size(
      image_size=input_size, filter_size=layer_temp.kernel_shape[0],
      padding=layer_temp.padding, stride=layer_temp.strides[0])
  # 3
  if block._has_se:
    se_reduce = getattr(block, '_se_reduce')
    se_expand = getattr(block, '_se_expand')
    # Kernel has the input features in its second dimension.
    n_channels = se_reduce.kernel.shape.as_list()[2]
    ops.append(('_se_reduce_mean', counting.GlobalAvg(input_size, n_channels)))
    # input size is 1
    layer_temp = conv_counter_class(
        1, se_reduce.kernel.shape.as_list(), se_reduce.strides,
        se_reduce.padding, True, f_activation)
    ops.append(('_se_reduce', layer_temp))
    layer_temp = conv_counter_class(
        1, se_expand.kernel.shape.as_list(), se_expand.strides,
        se_expand.padding, True, 'sigmoid')
    ops.append(('_se_expand', layer_temp))
    ops.append(('_se_scale', counting.Scale(input_size, n_channels)))

  # 4
  l_name = '_project_conv'
  layer = getattr(block, l_name)
  layer_temp = conv_counter_class(
      input_size, layer.kernel.shape.as_list(), layer.strides, layer.padding,
      True, None)  # Use bias true since batch_norm, no activation
  ops.append((l_name, layer_temp))

  if (block._block_args.id_skip
      and all(s == 1 for s in block._block_args.strides)
      and block._block_args.input_filters == block._block_args.output_filters):
    ops.append(('_skip_add', counting.Add(input_size, n_channels)))
  return ops, input_size


def read_model(model, input_shape, f_activation='swish'):
  """Reads the operations on a single EfficientNet block.

  Args:
    model: efficientnet_model.Model,
    input_shape: int, square image assumed.
    f_activation: str or None, one of 'relu', 'swish', None.

  Returns:
    list, of operations.
  """
  # Ensure that the input run through model
  _ = model(tf.ones(input_shape))
  input_size = input_shape[1]  # Assuming square
  ops = []
  # 1
  l_name = '_conv_stem'
  layer = getattr(model, l_name)
  layer_temp = counting.Conv2D(
      input_size, layer.weights[0].shape.as_list(), layer.strides,
      layer.padding, True, f_activation)  # Use bias true since batch_norm
  ops.append((l_name, layer_temp))
  # Input size might have changed.
  input_size = counting.get_conv_output_size(
      image_size=input_size, filter_size=layer_temp.kernel_shape[0],
      padding=layer_temp.padding, stride=layer_temp.strides[0])

  # Blocks
  for idx, block in enumerate(model._blocks):
    block_ops, input_size = read_block(block, input_size,
                                       f_activation=f_activation)
    ops.append(('block_%d' % idx, block_ops))

  # Head
  l_name = '_conv_head'
  layer = getattr(model, l_name)
  layer_temp = counting.Conv2D(
      input_size, layer.weights[0].shape.as_list(), layer.strides,
      layer.padding, True, f_activation)  # Use bias true since batch_norm
  n_channels_out = layer.weights[0].shape.as_list()[-1]
  ops.append((l_name, layer_temp))

  ops.append(('_avg_pooling', counting.GlobalAvg(input_size, n_channels_out)))

  return ops

F_ACTIVATION = 'swish'

all_ops = read_model(model, input_shape, f_activation=F_ACTIVATION)
from pprint import pprint
for op in all_ops:
    pprint(op)

"""## 3. Counting
- Let's define some constants need for counting.
  - `INPUT_BITS` used for the inputs of the multiplication.
  - `ACCUMULATOR_BITS` used for the accumulator of the additions.
  - `PARAMETER_BITS` used to store individual parameter: which is equal to `INPUT_BITS` for simplicity here.
  - `IS_DEBUG`, if True, reports the individual operations of a single block in addition to aggregations.
- Sparsity is applied on convolutional layers and fully connected layers and reduces number of multiplies and adds of a vector product.
- Sparsity mask is defined as a binary mask and added to the total parameter count.
"""

# add_bits_base=32, since 32 bit adds count 1 add.
# mul_bits_base=32, since multiplications with 32 bit input count 1 multiplication.
counter = counting.MicroNetCounter(all_ops, add_bits_base=32, mul_bits_base=32)

# Constants
INPUT_BITS = 16
ACCUMULATOR_BITS = 32
PARAMETER_BITS = INPUT_BITS
SUMMARIZE_BLOCKS = False

counter.print_summary(0, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=SUMMARIZE_BLOCKS)
