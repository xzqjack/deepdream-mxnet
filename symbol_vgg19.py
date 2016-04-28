import find_mxnet
import mxnet as mx

def get_symbol(input_size, dev):
    # declare symbol
    data = mx.sym.Variable("data")
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    out = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
#    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
#    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
#    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
#    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
#    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
#    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
#    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
#    conv4_4 = mx.symbol.Convolution(name='conv4_4', data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    relu4_4 = mx.symbol.Activation(name='relu4_4', data=conv4_4 , act_type='relu')
#    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
#    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
#    out = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')


    # make executor
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=input_size)
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]))
    grad_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]))  
    # init with pretrained weight
    pretrained = mx.nd.load("vgg19_80M.params")
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        pretrained[key].copyto(arg_dict[name])
    executor = out.bind(ctx=dev, args=arg_dict, args_grad=grad_dict, grad_req="write")
    return executor
