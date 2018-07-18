#include <vector>
#include <iostream>
#include "caffe/layers/convInt8_withKL_layer.hpp"
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
void ConvInt8withKLLayer<Dtype>::computeInt8Weight(int *idx,const Dtype t1,const Dtype t2)
{
  Dtype t1_pos = t1>0?t1:(-t1);
  Dtype t2_pos = t2>0?t2:(-t2);
  Dtype each_seg = (t2-t1);
  int this_idx = (*idx-this->preTestBatches);
  int whichWeightSeg=this_idx%(weight_adjust_segment_count*weight_adjust_each_count)/weight_adjust_each_count;
  if(current_weight_adjust_segment_idx != whichWeightSeg)
  {
    current_weight_adjust_segment_idx = whichWeightSeg;
    Dtype this_t = t1_pos+each_seg*whichWeightSeg/input_adjust_segment_count;
    this->blobs_[0].get()->mutable_cpu_data()[4]=this_t;
    this->blobs_[0].get()->mutable_cpu_data()[5]=this_t;
    this->weight_temp_unit_sacle = this_t/127;
    this->weight_temp_unit_sacle_1 = 127.0/this_t;
    
    TODO:
  }
  LOG(INFO)<<"weight_temp_unit_sacle = "<<this->weight_temp_unit_sacle<<";  weight_temp_unit_sacle_1 = "<<this->weight_temp_unit_sacle_1;
}
template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::computeInt8input(int *idx,const Dtype t1,const Dtype t2)
{
  Dtype t1_pos = t1>0?t1:(-t1);
  Dtype t2_pos = t2>0?t2:(-t2);
  Dtype each_seg = (t2-t1);
  int this_idx = (*idx-this->preTestBatches);
  int whichInputSeg=this_idx/(weight_adjust_segment_count*weight_adjust_each_count)/input_adjust_each_count;
  Dtype this_t = t1_pos+each_seg*whichInputSeg/input_adjust_segment_count;
  this->blobs_[0].get()->mutable_cpu_data()[2]=this_t;
  this->blobs_[0].get()->mutable_cpu_data()[3]=this_t;

  this->input_temp_unit_sacle = this_t/127;
  this->input_temp_unit_sacle_1 = 127.0/this_t;

  LOG(INFO)<<"input_temp_unit_sacle = "<<this->input_temp_unit_sacle<<";  input_temp_unit_sacle_1 = "<<this->input_temp_unit_sacle_1;
}

template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::weight2int8(const int count, const Dtype*fp32weights, signed char*int8weight, const Dtype minT, const Dtype maxT, const Dtype unit_scale, const Dtype bias, bool doBias)
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
void ConvInt8withKLLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const signed char* weights, Dtype* output, bool skip_im2col) {
      
  const signed char* col_buff;

  if (!is_1x1_) {
          CHECK( num_spatial_axes_ == 2);
          // std::cout<<"----------------------input----------------------"<<std::endl; // showDevice3(input,10); // std::cout<<"----------------------initi  col_buffer_----------------------"<<std::endl; // showDevice3(col_buffer_.gpu_data(),50); // printf("-----   %p\n",col_buffer_.mutable_gpu_data()); // printf("-----   %p\n",col_buffer_.gpu_data());
          im2col_gpu_quantized(input, conv_in_channels_, conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                                                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                                                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                                                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                                                    dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buffer_.mutable_gpu_data(),(Dtype)-0.4,(Dtype)0.4,(Dtype)317.5);
    col_buff = col_buffer_.gpu_data();
  }
  else
  {
        im2col_1x1_gpu_quantized(col_buffer_.count(), input, col_buffer_.mutable_gpu_data(), (Dtype)-0.4, (Dtype)0.4, (Dtype)317.5);
        col_buff = col_buffer_.gpu_data();
  }

  for (int g = 0; g < group_; ++g) {
    caffe_gpu_iGemm(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        1, weights + weight_offset_ * g, col_buff + col_offset_ * g,
        0, int32out.mutable_gpu_data() + output_offset_ * g);
  }

showDevice3(int32out.gpu_data(),50);
  int2Dtype(int32out.count(),int32out.gpu_data(),output,(Dtype)0.001);
  std::cout<<"----------------------output----------------------"<<std::endl; 
  showDevice3(output,50); 
  
#ifdef SHOW_FP32_OUT
   if (!is_1x1_) 
   {
        im2col_gpu(input, conv_in_channels_, conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                                                  kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                                                  pad_.cpu_data()[0], pad_.cpu_data()[1],
                                                  stride_.cpu_data()[0], stride_.cpu_data()[1],
                                                  dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buffer_show_.mutable_gpu_data());
   }
#endif

}
template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias)
{}


template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    if(!weightFp32HasExtracted)  
    {
        getFp32Weight();
        getMaxAndMIn(weightFp32.count(),weightFp32.gpu_data(), &(this->maxAndMin.mutable_gpu_data()[3]),&(this->maxAndMin.mutable_gpu_data()[2]));
        LOG(INFO)<<this->maxAndMin.cpu_data()[3]<<"\t\t"<<this->maxAndMin.cpu_data()[2];
    }
    if(preTestIdx<preTestBatches)
    {
        getMaxAndMIn(bottom[0]->count(),bottom[0]->gpu_data(), &(this->maxAndMin.mutable_gpu_data()[1]),&(this->maxAndMin.mutable_gpu_data()[0]));
        preTestIdx++;
        if(!isFirstGetMaxMin)
        {
            input_scale_t1=this->maxAndMin.cpu_data()[0];
            input_scale_t2=this->maxAndMin.cpu_data()[1];
            isFirstGetMaxMin = true;
        }
        else
        {
          this->maxAndMin.mutable_cpu_data()[0] = this->maxAndMin.cpu_data()[0]>input_scale_t1?input_scale_t1:this->maxAndMin.cpu_data()[0];
          this->maxAndMin.mutable_cpu_data()[1] = this->maxAndMin.cpu_data()[1]<input_scale_t2?input_scale_t2:this->maxAndMin.cpu_data()[1];
          input_scale_t1=this->maxAndMin.cpu_data()[0];
          input_scale_t2=this->maxAndMin.cpu_data()[1];
        }
        LOG(INFO)<<"input  region : "<<this->maxAndMin.cpu_data()[1]<<"\t\t"<<this->maxAndMin.cpu_data()[0];
        LOG(INFO)<<"weight region : "<<this->maxAndMin.cpu_data()[3]<<"\t\t"<<this->maxAndMin.cpu_data()[2];
        LOG(INFO)<<preTestIdx<<" < "<<preTestBatches;
    }
    if(preTestIdx<preTestBatches) return;
    computeInt8Weight(&preTestIdx,this->maxAndMin.cpu_data()[2],this->maxAndMin.cpu_data()[3]);
    computeInt8input(&preTestIdx,this->maxAndMin.cpu_data()[0],this->maxAndMin.cpu_data()[1]);
    preTestIdx++;
    return;
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
void ConvInt8withKLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(INFO)<<2;
    LOG(FATAL)<<"NOT IMPLEMENTED";
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvInt8withKLLayer);

}  // namespace caffe
