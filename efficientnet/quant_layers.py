from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D, BatchNormalization
from tensorflow.python.ops import init_ops


def quantize_activations(
        x,
        training,
        quantize,
        quantize_bits=7,
        min_deviation_multiplier=6,
        max_deviation_multiplier=6
):

    batchnorm = BatchNormalization(
        name='quantization_batchnorm',
        center=False,
        scale=False
    )
    normalized_x = batchnorm(x, training=training)

    x = x + normalized_x * 0

    deviation = tf.sqrt(batchnorm.moving_variance)
    range_min = tf.reduce_mean(batchnorm.moving_mean - min_deviation_multiplier * deviation)
    range_max = tf.reduce_mean(batchnorm.moving_mean + max_deviation_multiplier * deviation)

    #tf.summary.histogram(
    #    name = 'activations',
    #    values = x,
    #    family = 'activations'
    #)

    #tf.summary.histogram(
    #    name='max_range_dist',
    #    values=range_max,
    #    family='max_ranges'
    #)

    #tf.summary.histogram(
    #    name = 'min_range_dist',
    #    values = range_min,
    #    family = 'min_ranges'
    #)

    #tf.summary.histogram(
    #    name='max_range_dist',
    #    values=range_max,
    #    family='max_ranges'
    #)

    if not training:
        if quantize:
            return tf.quantization.fake_quant_with_min_max_vars(
                x,
                min=range_min,
                max=range_max,
                num_bits=quantize_bits
            )
    return x


@tf.custom_gradient
def sign(x):
    return (tf.cast(x > 0, tf.float32) * 2 - 1), lambda dy: dy


@tf.custom_gradient
def sign_w_alpha(x):
    alpha = tf.reduce_mean(tf.math.abs(x))
    return (tf.cast(x > 0, tf.float32) * 2 - 1) * alpha, lambda dy: dy


def _fake_cast_float16(input):
    return tf.cast(tf.cast(input, tf.float16), tf.float32)

def _float16_cast(input_):
    return tf.cast(input_, tf.half)

class BinConv2D(Conv2D):

    def __init__(
            self, filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=init_ops.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            binarize_activations=False,
            use_alpha=False,
            quantize=False,
            quantize_bits=8,
            min_deviation_multiplier=6,
            max_deviation_multiplier=6,
            **kwargs
    ):
        super(BinConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs)

        assert data_format == 'channels_last'

        self._binarize_activations = binarize_activations
        self._use_alpha = use_alpha
        self._quantize = quantize
        self._quantize_bits = quantize_bits
        self._min_deviation_multiplier = min_deviation_multiplier
        self._max_deviation_multiplier = max_deviation_multiplier

    def call(self, inputs, training):

        if self._use_alpha:
            alpha = tf.reduce_mean(tf.math.abs(self.kernel))
            bin_kernel = tf.cast(self.kernel > 0, tf.float32) * 2 - 1
            # bin_kernel = sign_w_alpha(self.kernel)
            inputs = inputs * alpha
        else:
            bin_kernel = sign(self.kernel)

        inputs = quantize_activations(
            inputs,
            training=training,
            quantize=self._quantize,
            quantize_bits=self._quantize_bits,
            min_deviation_multiplier=self._min_deviation_multiplier,
            max_deviation_multiplier=self._max_deviation_multiplier
        )

        bin_kernel = bin_kernel

        outputs = self._convolution_op(inputs, bin_kernel)

        if self.use_bias:
            # only channels_last format
            self.bias = _fake_cast_float16(self.bias)
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        #outputs = tf.cast(outputs, tf.float32)

        if self._binarize_activations:
            outputs = sign(outputs)
        else:
            if self.activation is not None:
                return self.activation(outputs)

        return outputs


class Conv2D(tf.layers.Conv2D):
    # pass
    def call(self, inputs):
        inputs = _fake_cast_float16(inputs)

        quant_kernel = _fake_cast_float16(self.kernel)

        outputs = self._convolution_op(_float16_cast(inputs), _float16_cast(quant_kernel))
        outputs = tf.cast(outputs, tf.float32)

        if self.use_bias:

            bias = _fake_cast_float16(self.bias)

            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, bias, data_format='NHWC')

        #outputs = _fake_cast_float16(outputs)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    def call(self, inputs, training=None):
        inputs = _fake_cast_float16(inputs)
        quant_kernel = _fake_cast_float16(self.depthwise_kernel)
        outputs = backend.depthwise_conv2d(
            _float16_cast(inputs),
            _float16_cast(quant_kernel),
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)
        outputs = tf.cast(outputs, tf.float32)

        if self.use_bias:
            bias = _fake_cast_float16(self.bias)
            outputs = backend.bias_add(
                outputs,
                bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            return self.activation(outputs)

        return outputs
