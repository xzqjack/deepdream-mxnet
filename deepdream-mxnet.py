# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import PIL.Image
import scipy.ndimage as nd
from skimage import io,transform
import mxnet as mx
import os

dev = mx.gpu(0)
img_mean = np.array([[[ 104.]],[[ 116.]],[[ 122.]]])

    

def preprocess(img,img_mean):
    return np.float32(np.rollaxis(img, 2)[::-1]) - img_mean
    
def deprocess(img,img_mean):
    return np.uint8(np.clip(np.dstack((img + img_mean)[::-1]), 0, 255))

def objective_function(data,loss_function,guide=None):
    if loss_function == "L2":
        grad = data
    elif loss_function == "inner_product":
        ch = data.shape[0]
        data = data.reshape(ch,-1)
        guide = guide.reshape(ch,-1)
        A = data.T.dot(guide) # compute the matrix of dot-products with guide features
        grad = guide[:,A.argmax(1)] # select ones that match best       
    return grad     
    
def make_step(executor, input_img, img_mean, loss_function, guide,step_size=2.5,jitter=32, clip=True):
    '''Basic gradient ascent step.'''
    h, w = input_img.shape[-2:]
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    input_img = np.roll(np.roll(input_img, ox, -1), oy, -2) # apply jitter shift
    input_img = input_img.reshape(1,3,h,w)
    executor.arg_dict["data"][:] = input_img
    executor.forward(is_train=True)
    feature_grad = mx.ndarray.zeros(executor.outputs[0].shape,ctx=mx.gpu(0))    
    data = executor.outputs[0].asnumpy().squeeze()    
    feature_grad[:] = objective_function(data,loss_function=loss_function,guide=guide).reshape(executor.outputs[0].shape)
    executor.backward(feature_grad)
    g = executor.grad_dict['data'].asnumpy()
    # apply normalized ascent step to the input image
    input_img += step_size/np.abs(g).mean() * g        
    input_img = np.roll(np.roll(input_img.reshape(3,h,w), -ox, -1), -oy, -2) # unshift image                    
    if clip:
        bias = img_mean
        input_img[:] = np.clip(input_img, -bias, 255-bias)    
    return input_img
    
def deepdream(executor, base_img, img_mean, loss_function, guide_input=None,
              dev=mx.gpu(0),iter_n=10, octave_n=4, octave_scale=1.4,clip=True):
    # prepare base images for all octaves
    octaves = [preprocess(base_img,img_mean)] 
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)
        executor = executor.reshape(data=(1,3,h,w),allow_up_sizing=True) # resize the network's input image size
        if guide_input is not None:
            octave_guide = transform.resize(guide_input,(h,w))*256
            octave_guide = preprocess(octave_guide,img_mean)
            executor.arg_dict["data"][:] = octave_guide .reshape(1,3,h,w)
            executor.forward()
            guide = executor.outputs[0].asnumpy().squeeze()
        input_img = (octave_base+detail)
        for i in xrange(iter_n):
            input_img = make_step(executor, input_img, img_mean, loss_function=loss_function, 
                                  guide=guide, clip=clip)
            # visualization
            vis = deprocess(input_img,img_mean)
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
#            showarray(vis)
            print octave, i, vis.shape
            io.imsave('output/octave_'+str(octave)+'iteration_'+str(i)+'.jpg',vis)
#            clear_output(wait=True)
        # extract details produced on the current octave
        detail = input_img - octave_base
    # returning the resulting image
    return deprocess(input_img,img_mean)

img = io.imread('input_image/sky1024px.jpg')
img = transform.resize(img,(img.shape[0],img.shape[1]))*256
input_size = (1,3,img.shape[0],img.shape[1])
import importlib
vgg_executor = importlib.import_module('Inception-V3').get_symbol(input_size, dev)
#_=deepdream(vgg_executor, img, img_mean=img_mean, loss_function="L2")


    
#make frame
#frame_file = 'frame'
#if not os.path.exists(frame_file):
#    os.mkdir(frame_file)
#frame = img
#frame_i = 0
#h, w = frame.shape[:2]
#s = 0.05 # scale coefficient
#for i in xrange(100):
#    frame = deepdream(vgg_executor, frame,img_mean=img_mean)
#    PIL.Image.fromarray(np.uint8(frame)).save(frame_file+"/%04d.jpg"%frame_i)
#    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
#    frame_i += 1

#Controlling dreams
guide_input = io.imread('input_image/flowers.jpg')
_=deepdream(vgg_executor, img, img_mean=img_mean, loss_function="inner_product",guide_input=guide_input)
