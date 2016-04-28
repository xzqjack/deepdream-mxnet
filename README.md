# deepdream-mxnet by xzqjack
该python代码对[deepdream](https://github.com/google/deepdream)进行了mxnet移植，实现了[deepdream](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html)功能。
目前仅提高三种模型，分别是Inception-V3,VGG16,VGG19。相关模型参数可以去[inception-V3](https://github.com/dmlc/mxnet-model-gallery)下载，或者安装mxnet后用mxnet提供的转换工具转换caffe模型。
更多模型定义和参数请参考[mxnet官方网站](https://github.com/dmlc/mxnet)。


在此给大家安利下mxnet，它的好，谁用谁知道。

希望更多的科研、工作因为使用mxnet变得更简单高效！

***

This repository contains python code to reimplement the [deepdream](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html) with **mxnet**.
There are 3 model.py defining cnn model, Inception-V3,VGG16,VGG19, You can download from the parameters from [inception-V3](https://github.com/dmlc/mxnet-model-gallery) parameters or convert the  caffe vgg-model into mxnet model with convert.py in [mxnet/tools](https://github.com/dmlc/mxnet/tree/master/tools/caffe_converter).

Here are some results
Inception-V3
![Inception-V3](https://github.com/xzqjack/deepdream-mxnet/blob/master/output/Inception-V3.jpg)
Inception-V3 with flower
![Inception-V3 with flower](https://github.com/xzqjack/deepdream-mxnet/blob/master/output/Inception-V3%20with%20flower.jpg)
VGG-16
![vgg16-pool5](https://github.com/xzqjack/deepdream-mxnet/blob/master/output/vgg16-pool5.jpg)
vgg16-pool5 with flower
![vgg16-pool5 with flower](https://github.com/xzqjack/deepdream-mxnet/blob/master/output/vgg16-pool5%20with%20flower.jpg)
VGG-19
![vgg19-relu5_1](https://github.com/xzqjack/deepdream-mxnet/blob/master/output/vgg16-pool5.jpg)
vgg19-relu5_1 with flower
![vgg19-relu5_1 with flower](https://github.com/xzqjack/deepdream-mxnet/blob/master/output/vgg19%20with%20flower.jpg)


You can generate more dream picture with [mxnet](http://mxnet.readthedocs.io/en/latest/) which is a high  Flexible and efficient Library for deeplearning.The installation of mxnet is very simple.This code was mainly referencing [deepdream](https://github.com/google/deepdream) released by [Alexander Mordvintsev](mailto:moralex@google.com) with caffe. However, it's still strongly recommended to try deepdream with mxnet because the installation of caffe maybe very hard for most people.

Thanks for the researchers' work!

注：目前inception-v3只能可视化in3c之前的层，后面的层会报错，已告诉mxnet开发者。修复后会在这里再次声明。

