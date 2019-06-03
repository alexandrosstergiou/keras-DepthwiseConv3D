# Depthwise 3DConvolutions in Keras

An extension of separable convolutions for 3D volumes. Performs volumetric convolutions for each channel of the input volume and will increase the output volume based on the number of convolutional operations (denoted as `depth_multiplier` inside the code). Current version only supports `channel_first` volumes as inputs.

*Base code for the implementation is used from: https://github.com/titu1994/MobileNetworks/blob/master/depthwise_conv.py*


<img src="https://github.com/alexandrosstergiou/keras-DepthwiseConv3D/images/depthwise_3dconv.png?raw=true" height=100% width=100%>

## Requirements

* Keras 2.2.4+
* Tensorflow 1.13

## Usage
```python
from DepthwiseConv3D import DepthwiseConv3D

input = Input(...)

x = DepthwiseConv3D(kernel_size=(3,3,3), depth_multiplier=2)(input)
...
```

## References:

[1] F. Chollet, Xception: Deep Learning with Depthwise Separable Convolutions [[link]](https://arxiv.org/pdf/1610.02357.pdf)
