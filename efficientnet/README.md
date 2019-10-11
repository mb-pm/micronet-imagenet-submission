# Micronet ImageNet Submission


## Training on TPUs.


To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource
   * A GCS bucket to store your training checkpoints (the "model directory")
   * Install TensorFlow version >= 1.13 for both GCE VM and Cloud.

Then train the model:


    $ bash run_train.sh <TPU_NAME> <DATA_DIR> <MODEL_DIR>

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access

For more instructions, please refer to our tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet

## Evaluation on TPUs.

    $ bash run_eval.sh <DATA_DIR> <MODEL_DIR>

    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access

Evaluation is run on CPU/GPU.

## Approach

The starting point of our approach was EfficientNet_B3 ([paper](https://arxiv.org/pdf/1905.11946.pdf)). Its main
building block is mobile inverted bottleneck block (MBConv), to which authors also add squeeze-and-excitation optimization.
In total, EfficientNet_B3 has one Conv2D, as the first layer, followed by 26 MBConv blocks and it ends with
Conv1x1, GlobalAveragePooling and FC layer with Softmax activation.


Original model: EfficientNet_B3

    Conv3x3
    MBConv0
    MBConv1
    MBConv2
    ...
    MBConv24
    MBConv25
    Conv1x1
    GlobalAveragePooling
    FC

The described model achieves 81.1% Top-1 Accuracy on Imagenet, which gave us some space for optimization.

Our first step was binarization according to Binary Connect [paper](https://arxiv.org/pdf/1511.00363.pdf).
We binarized weights in 4 out of 5 convolutional layers in MBConv block (binarized blocks \[2, 25\]).
Binarization significantly reduced the number of parameters and FLOPS.

The second step was to remove FC layer at the end of the model, because almost half of the parameters count was stored in FC weights.

The third step was to quantize the ([reference](https://www.tensorflow.org/lite/performance/post_training_quantization)) inputs of
binarized convolutional layers in MBConvBlockBinary blocks which significantly reduced operations and parameters as well. We managed to quantize these
layers to less than 8 bits. Training of the model is quantization aware (during the train process model calculates min/max statistics
which we apply in the evaluation process using TensorFlow's fake quantization nodes). All the other convolutional layers (their weights
and activations) were quantized to 16 bits using simple cast to float16 and TensorFlow convolution operates with float16 if float16 inputs/activations are provided (Used implementations: [conv2d_gpu_float16](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_2d_gpu_half.cu.cc), [depthwise_conv2d_gpu_float16](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/depthwise_conv_op_gpu_half.cu.cc)).

Submission model: **QuantEfficientNet_B3**

    Conv3x3                         (weight/activations quantization 16)
    MBConv0                         (weight/activations quantization 16)
    MBConv1                         (weight/activations quantization 16)
    MBConvBlockBinary2     (weight quantization 1, activation quantization 5 or 8 bits)
    ...                                     (weight quantization 1, activation quantization 5 or 8 bits)
    MBConvBlockBinary24  (weight quantization 1, activation quantization 5 or 8 bits)
    MBConvBlockBinary25  (weight quantization 1, activation quantization 5 or 8 bits)
    Conv1x1                         (weight/activations quantization 16)
    GlobalAveragePooling


The hyperparameters for training of the model are as described in train scripts ([official implementation repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)) with batch size being set to `1024`,
optimizer being `RMSprop` with learning rate `0.005` which decreases by `0.97` every `2.4` epochs.


## Counting

Here is information about how we counted parameters and operations for layers in our submissions.
In all counting formulas, H represents image/feature map height, W represents image/feature map height, C
represents number of image/feature map channels.

### Convolutional layers
Since all our convolutional layers are binary with stride 1 and without biases,
calculation is as follows:

* BinConv2D (all layers use stride 1)
    * parameters
        * parameters are counted as 1/32 of parameter size since they can be represented in 1 bit
        * number of parameters per layer is K\*K\*C<sub>in</sub>\*C<sub>out</sub>/32, where K is kernel size,
    C<sub>in</sub> is number of input channels, and C<sub>out</sub> is number of output channels

    * operations

        * multiplications
            * total number of multiplications is scaled with 1/32 since the multiplications
            can be carried by modifying single bit
            * i.e. for convolution over HxWxC<sub>in</sub> feature map
            and kernel size KxK, number of multiplications is K\*K\*C<sub>in</sub>\*N/32, where N is the number of
            output feature map elements (W\*H\*C<sub>out</sub>)

        * additions
            * since the multiplication operation is just changing the single bit (multiplication with -1 or 1),
            and that does not change precision of the input element (even in the
            kernels implemented in deep learning frameworks, so rounding the numbers to wanted precision is not needed
            after the multiplication operation), we assume that total number of additions is scaled with n/32,
            where n is the input bitwidth to convolutional layer - in our submission, bitwidth of inputs
            is 5 for all layers in MBConvBlockBinary except the last layers which has bitwidth 8.
            * i.e. for convolution over HxWxC<sub>in</sub> feature map
            and kernel size KxK, number of additions is (K\*K\*C<sub>in</sub>-1)\*N*n/32, where N is the number of
            output feature map elements (W\*H\*C<sub>out</sub>), and n is the input bitwidth

* Conv2D
    * parameters
        * parameters are counted as 16/32 of parameter size since we cast them to float16
        * number of parameters per layer is K\*K\*C<sub>in</sub>\*C<sub>out</sub>*16/32, where K is kernel size,
    C<sub>in</sub> is number of input channels, and C<sub>out</sub> is number of output channels

    * operations

        * multiplications
            * total number of multiplications is scaled with 16/32
            * i.e. for convolution over HxWxC<sub>in</sub> feature map
            and kernel size KxK, number of multiplications is K\*K\*C<sub>in</sub>\*N*16/32, where N is the number of
            output feature map elements (W/s\*H/s\*C<sub>out</sub>, where `s` is kernel stride)

        * additions
            * total number of multiplications is scaled with 16/32
            * i.e. for convolution over HxWxC<sub>in</sub> feature map
            and kernel size KxK, number of multiplications is (K\*K\*C<sub>in</sub>-1)\*N*16/32, where N is the number of
            output feature map elements (W/s\*H/s\*C<sub>out</sub>, where `s` is kernel stride)


* DepthwiseConv2D
    * parameters
        * parameters are counted as 16/32 of parameter size since we cast them to float16
        * number of parameters per layer is K\*K\*C<sub>in</sub>\*16/32, where K is kernel size,
    C<sub>in</sub> is number of input channels

    * operations

        * multiplications
            * total number of multiplications is scaled with 16/32
            * i.e. for convolution over HxWxC<sub>in</sub> feature map
            and kernel size KxK, number of multiplications is K\*K\*C<sub>in</sub>\*N*16/32, where N is the number of
            output feature map elements (W/s\*H/s, where `s` is kernel stride)

        * additions
            * total number of multiplications is scaled with 16/32
            * i.e. for convolution over HxWxC<sub>in</sub> feature map
            and kernel size KxK, number of multiplications is (K\*K-1)\*C<sub>in</sub>\*N*16/32, where N is the number of
            output feature map elements (W/s\*H/s\, where `s` is kernel stride)


### BatchNorm layers
BatchNorm layers are used only after each convolutional layer.
Since we use binary weighted convolutional layers, BatchNorm layers cannot be fused into them. However,
we fuse BatchNorm layer into single linear transformation (single multiply and add per input element).

* parameters of BatchNorm layers are not quantized in any way - they are in full, float32 format
    * number of parameters per layer is 2 * C where C is the number of channels
    of feature map on which BatchNorm is performed
* operations
    * multiplications
        * total number of multiplications performed equals to the number of total input elements in feature map -
        H\*W\*C
    * additions
        * total number of additions performed equals to the number of total input elements in feature map -
        H\*W\*C


### GlobalAveragePool layers
GlobalMaxPool layer is, in our case, used after all convolutional layers and it's the last layer in the model before Softmax.
It is parameter-less, but it (similarly as MaxPool layers) costs multiplication
operations (comparisons).

* operations
    * multiplications
        * for every channel we divide the sum of channel values with number of elements - total number is C were C is number of channels.
    * additions
        * total number of additions is (H\*W-1)\*C


### Activation layer (Swish and Sigmoid)

* Sigmoid
    * operations
        * multiplications
            * based on official script for counting FLOPS Sigmoid has 2 multiply ops
        * additions
            * based on official script for counting FLOPS Sigmoid has 1 addition op

* Swish
    * operations
        * multiplications
            * based on official script for counting FLOPS Sigmoid has 3 multiply ops
        * additions
            * based on official script for counting FLOPS Sigmoid has 1 addition op

## Quantization method used in block MBConvBlockBinary in BinConv2D layer
We used TensorFlow's fake quantization (tf.quantization.fake_quant_with_min_max_vars). Number of bits used for fake quantization is 5 for all layers in
MBConvBlockBinary except the last layers which has bitwidth 8.

Fake quantization layer requires range in which the quantization is performed. We calculated mean and variance of activations,
on the train set, for every quantized layer and by visualizing histograms of the activations we determined min and max.
(e.g. min = mean - 6\*deviations, max = mean + 6\*deviations).
Since we want to bucket our activation in range \[min, max\] (per layer) into 256 different values. Quantization is performed in
following manner:

1) We want to make sure that after quantization, we have number representation which allows 1 bit multiplication
in our binary convolutions and fast, low-bitwidth multiplication in our binary convolutions
2) We want to make sure that quantized values have free conversion from/to full precision float32 values

Following are all theoretical assumptions - ideally, we would have something like signed magnitude representation
of sign + absolute value of quantized index and arithmetic unit that would optimally add those numbers. Additionally,
our binary 1 bit multiplication would also be feasible because of the sign bit.
To make sure we satisfy these properties, we would make the quantization/dequantization for the
inference in the following manner (that way of quantizing/dequantizing produce same float32 dequantized values as
does tf.quantization.fake_quant_with_min_max_vars layer - ordering is a bit different for our, theoretical inference
layer and TensorFlow quantization layer, but end results are the same):

* Quantization
    1) Fix \[min, max\] range same as TensorFlow does [documentation](https://www.tensorflow.org/api_docs/python/tf/quantization/fake_quant_with_min_max_vars)
    2) Clamp the input float32 value into `[min, max]` range
    3) Subtract the minimum value min from clamped float32 value
    4) Divide the value with step value - step value is equal to `(max-min)/(m-1)`, where m is total number of possible
    quantized values (256 in our case)
    5) Round the value to get index of quantized value (in range from 0 to 255)
    6) Mask the clamped value `n` exponential bits with bitstring index of quantized value
    7) Use only first `n+1` bits of the number as quantized value

When float32 values are quantized this way, we are using sign bit, and `n` exponent bits of full float32 number.
All other numbers of full float32 value are not needed, so we can just clamp first `n+1` bits of quantized value
(something like first 16 bits in bfloat16). Numbers quantized like that are still valid float numbers which can be used
in convolutional layers and satisfy properties mentioned earlier (1 bit multiplication with binary weights because of
the bit sign, free conversion to/from float32 by clamping/adding `32-n-1` bits). Convolution is carried on number quantized to
`n+1` bits - additions will also be in `n+1` bits because binary multiplication won't make any changes to quantized values (
except of the sign change).

* Dequantization
    1) Convert the number into the full float32 (by setting sign bit + `n` exponent bits)
    2) Shift full float32 number right to extract unsigned quantization index value
    3) Multiply the extracted value with the step value - step value is equal to `(max-min)/(m-1)`, where `m` is total number of possible
    quantized values(`m=2**n`)
    4) Multiply value from step 3) with the sign bit of value from step 1)
    4) Add the minimum value min to the multiplied value

* parameters
    * our quantization method has 3 parameters - min, max and step which are used in quantization and dequantization
* operations (all the operations are performed on full precision, float32 values)

    * quantization

        * multiplications (per value of feature map)
            * 2 multiplications for clamping the value into `[min, max]` range (2 comparisons)
            * 1 multiplication to divide the value with step value
            * 1 multiplication for round operation
            * 3 multiplications for masking the clamped value 7 exponential bits with quantization index
            * total = 7 multiplications per input element

        * additions (per value of feature map)
            * 1 addition for subtracting the minimum value from clamped value

    * dequantization

        * multiplications (per value of feature map)
            * 3 multiplications to extract quantization index value and calculate the step that needs to be
            added to minimum value

        * additions
            * 1 addition for adding the minimum value
    * in total, quantization + dequantization need 10 multiplications and 2 additions per input element
    * total number of multiplications performed equals to the multiply of total input elements in feature map and
     total number of multiplications per input element - H\*W\*C\*10
    * total number of additions performed equals to the multiply of total input elements in feature map and
     total number of additions per input element - H\*W\*C\*2

## Running the counter script
We took the original script for counting parameters and FLOPS for EfficientNet models from [here](
https://colab.research.google.com/github/google-research/google-research/blob/master/micronet_challenge/EfficientNetCounting.ipynb#scrollTo=P5p1fkA3rgL_),
and we modified it to account for our changes in the architecture. Our version of the script is located in following files:
`efficientnet/counting.py`, `efficientnet/efficient_net_counting.py`

To run the counting script position your self in `efficientnet/` directory and run:

    $ python3 efficient_net_counting.py

## Results


|         Name              | Accuracy | FLOPS score | Parameters score | Total   |
|:-------------------------:|:--------:|:--------------------:|:-------------------------:|:-------:|
| QuantEfficientNet_b3      |  75.59   |        0.52        |           0.1          |  0.62 |


The results for Flops, Parameters and Total Micronet Score are normalized against MobileNet_v2 as described
in the submission instructions.