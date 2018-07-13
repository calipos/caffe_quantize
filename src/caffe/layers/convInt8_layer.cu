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
    CHECK(minT < maxT);
    caffe_gpu_quantize_nobias(count, fp32weights, int8weight, minT, maxT, unit_scale);
  }
  else
  {
    LOG(FATAL)<<"NOT IMPELMENT!";
  }
}


template <typename Dtype>
void ConvInt8Layer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const signed char* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;

  if (!is_1x1_) {
    if (!skip_im2col) {

      conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());

    }
    col_buff = col_buffer_.gpu_data();
  }
showDevice(col_buffer_.gpu_data(),50);
showDevice(input,50);
  exit(0);
  weight2int8(col_buffer_.count(), col_buff, inputInt8.mutable_gpu_data(), this->blobs_[0]->gpu_data()[2], this->blobs_[0]->gpu_data()[3]);
  for (int g = 0; g < group_; ++g) {
    // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        // group_, conv_out_spatial_dim_, kernel_dim_,
        // (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        // (Dtype)0., output + output_offset_ * g);
  }
}
template <typename Dtype>
void ConvInt8Layer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias)
{}


template <typename Dtype>
void ConvInt8Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const signed char* weight = this->blobs_int8_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvInt8Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    LOG(FATAL)<<"NOT IMPLEMENTED";
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvInt8Layer);

}  // namespace caffe
