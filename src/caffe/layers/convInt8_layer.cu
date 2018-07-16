#include <vector>
#include <iostream>
#include "caffe/layers/convInt8_layer.hpp"
//#include "caffe/util/im2col.hpp"
namespace caffe {

using std::cout;
using std::endl;

template <typename Dtype>
void showDevice3(const Dtype*data,int count)
{
    Dtype *show=(Dtype*)malloc(count*sizeof(Dtype));
    cudaMemcpy(show,data,count*sizeof(Dtype),cudaMemcpyDeviceToHost);
    for(int i=0;i<count;i++)
    {
        std::cout<<(float)show[i]<<" ";
        if(i%10==9)std::cout <<std::endl;
    }
    free(show);
}


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
        CHECK(!force_nd_im2col_ && num_spatial_axes_ == 2);
// std::cout<<"----------------------input----------------------"<<std::endl;
// showDevice3(input,10);
// std::cout<<"----------------------initi  col_buffer_----------------------"<<std::endl;
// showDevice3(col_buffer_.gpu_data(),50);
// printf("-----   %p\n",col_buffer_.mutable_gpu_data());
// printf("-----   %p\n",col_buffer_.gpu_data());
        im2col_gpu_quantized(input, conv_in_channels_, conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                                                  kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                                                  pad_.cpu_data()[0], pad_.cpu_data()[1],
                                                  stride_.cpu_data()[0], stride_.cpu_data()[1],
                                                  dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buffer_.mutable_gpu_data(),(Dtype)-0.4,(Dtype)0.4,(Dtype)317.5);


      
      
      
        im2col_gpu(input, conv_in_channels_, conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                                                  kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                                                  pad_.cpu_data()[0], pad_.cpu_data()[1],
                                                  stride_.cpu_data()[0], stride_.cpu_data()[1],
                                                  dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buffer_show_.mutable_gpu_data());

                                                  
  exit(0);
    }
    //col_buff = col_buffer_.gpu_data();
  }


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
