"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
import find_mxnet
import mxnet as mx
from collections import namedtuple

generator_Executor = namedtuple('generator_Executor', ['executor','data', 'arg_dict', 'grad_dict','output'])
def get_symbol(input_size,dev):
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    out = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="out")
    # group 3
#    conv3_1 = mx.symbol.Convolution(
#        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
#    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
#    conv3_2 = mx.symbol.Convolution(
#        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
#    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
#    pool3 = mx.symbol.Pooling(
#        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
#    # group 4
#    conv4_1 = mx.symbol.Convolution(
#        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
#    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
#    conv4_2 = mx.symbol.Convolution(
#        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
#    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
#    pool4 = mx.symbol.Pooling(
#        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
#    # group 5
#    conv5_1 = mx.symbol.Convolution(
#        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
#    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
#    conv5_2 = mx.symbol.Convolution(
#        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
#    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
#    pool5 = mx.symbol.Pooling(
#        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
#    # group 6
#    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
#    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
#    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
#    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
#    # group 7
#    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
#    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
#    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
#    # output
#    fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=1000, name="fc8")
#    softmaxt = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=input_size)
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]))
    grad_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]))    
    # init with pretrained weight
    pretrained = mx.nd.load("vgg16.params")
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        pretrained[key].copyto(arg_dict[name])
    executor = out.bind(ctx=dev, args=arg_dict, args_grad=grad_dict, grad_req="write")
    return executor
