#coding=utf-8

import os.path as osp
import sys
import copy
import os
import numpy as np
import numpy.linalg as linalg

CAFFE_ROOT = '/home/lbl/lbl_trainData/git/test_git_quantize'
if osp.join(CAFFE_ROOT,'python') not in sys.path:
        sys.path.insert(0,osp.join(CAFFE_ROOT,'python'))

import caffe


caffe.set_mode_gpu()
nobias_net = caffe.Net("/media/hdd/lbl_trainData/git/test_git_quantize/examples/nvidia_quantize/nobias.prototxt", caffe.TEST)
stem1Layer = nobias_net._layer_by_name("stem1")
print len(stem1Layer.blobs)
print (stem1Layer.blobs[0].shape_string())
print (stem1Layer.blobs[0].data[0])
# print (stem1Layer.blobs[1].data[0])
# print (stem1Layer.blobs[1].data[1])
nobias_net.save("nobias.caffemodel")
exit(0)

caffe.set_mode_gpu()
original_net = caffe.Net('./quantizedNet.prototxt', caffe.TEST)
original_net.blobs['data'].data[...] = np.random.random((1,3,304,304))-0.5
original_net.forward()

print "---------"
exit(0)
original_net.save("quantized_.caffemodel")
exit(0)
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