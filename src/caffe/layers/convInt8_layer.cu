#include <vector>
#include <iostream>
#include "caffe/layers/convInt8_layer.hpp"

namespace caffe {

using std::cout;
using std::endl;


template <typename Dtype>
void ConvInt8Layer<Dtype>::weight2int8(const int count, const Dtype*fp32weights, signed char*int8weight, const Dtype minT, const Dtype maxT, const Dtype unit_scale, const Dtype bias, bool doBias)
{
  Dtype weight_uni_scale=0;
  if(unit_scale>0)
  {weight_uni_scale=unit_scale;}
  else
  {weight_uni_scale=(maxT-minT)/255;}
  if(!doBias)
  {
    caffe_gpu_quantize_nobias(count, fp32weights, int8weight, minT, maxT, unit_scale);
  }
  else
  {
    LOG(FATAL)<<"NOT IMPELMENT!";
  }



}

template <typename Dtype>
void ConvInt8Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


}

template <typename Dtype>
void ConvInt8Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(ConvInt8Layer);

}  // namespace caffe
