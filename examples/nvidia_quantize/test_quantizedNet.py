#coding=utf-8

import os.path as osp
import sys
import copy
import os
import numpy as np
import numpy.linalg as linalg

CAFFE_ROOT = '/home/lbl/lbl_trainData/git/caffe_nvidia_quantize'
if osp.join(CAFFE_ROOT,'python') not in sys.path:
        sys.path.insert(0,osp.join(CAFFE_ROOT,'python'))

import caffe

caffe.set_mode_gpu()
original_net = caffe.Net('./quantizedNet.prototxt', caffe.TEST)
im = np.random.random((1,3,304,304))-0.5

data_original_net = original_net.blobs['data']
data_original_net.data[...] = im
original_net.forward()

original_net.save("quantized_.caffemodel")

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
caffe.set_mode_gpu()
quantized_net = caffe.Net('./quantizedNet.prototxt', "./quantized_.caffemodel", caffe.TEST)
layer_q = quantized_net._layer_by_name("stem1")
print "layer_q.blobs.size() = ",len(layer_q.blobs)
print "layer_q.blobs[0] = ",(layer_q.blobs[0].shape_string())
print "layer_q.blobs[1] = ",(layer_q.blobs[1].shape_string())
print "layer_q.int8blobs.size() = ",layer_q.int8blobssize()
print "layer_q.int8blobs[0] = ",layer_q.int8blobsshapestring(0)
print "show int8 : ",layer_q.showint8blobs(0,5)
quantized_net.blobs['data'].data[...] = im
quantized_net.forward()