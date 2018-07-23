cd /media/hdd/lbl_trainData/git/caffe_nvidia_quantize
./build/tools/caffe train \
--solver="/media/hdd/lbl_trainData/git/caffe_nvidia_quantize/examples/nvidia_quantize/solver.prototxt" \
--weights="/media/hdd/lbl_trainData/git/caffe_nvidia_quantize/examples/nvidia_quantize/nobias.caffemodel" \
--gpu 0 2>&1 | tee /media/hdd/lbl_trainData/git/caffe_nvidia_quantize/examples/nvidia_quantize/log.log
