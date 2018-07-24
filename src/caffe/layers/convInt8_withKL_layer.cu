#include <vector>
#include <iostream>
#include "caffe/layers/convInt8_withKL_layer.hpp"
//#include "caffe/util/im2col.hpp"
namespace caffe {

using std::cout;
using std::endl;


int getNewDim(int n, int k,int*newN,int*newK)
{
	if(n%4==0 && k%4==0)
	{
		*newN = n;
		*newK = k;
		return 0;
	}	
	*newN = n%4==0? n : (n/4+1)*4;
	*newK = k%4==0? k : (k/4+1)*4;
	return 1;
}


template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::computeInt8Weight(int *idx,const Dtype t1,const Dtype t2)
{
  Dtype t1_pos = t1>0?t1:(-t1);
  //Dtype t2_pos = t2>0?t2:(-t2);
  Dtype each_seg = (t2-t1);
  int this_idx = (*idx-this->preTestBatches);
  int whichWeightSeg=this_idx%(weight_adjust_segment_count*weight_adjust_each_count)/weight_adjust_each_count;
  if(current_weight_adjust_segment_idx != whichWeightSeg)
  {
    current_weight_adjust_segment_idx = whichWeightSeg;
    Dtype this_t = t1_pos+each_seg*whichWeightSeg/input_adjust_segment_count;
    this->blobs_[0].get()->mutable_cpu_data()[4]=this_t*-1;
    this->blobs_[0].get()->mutable_cpu_data()[5]=this_t;
    this->weight_temp_unit_sacle = this_t/127;
    this->weight_temp_unit_sacle_1 = 127.0/this_t;
#ifdef SHOW_WEIGHT2INT8
    std::cout<<"--------weight_T : "<<this_t<<"--------"<<std::endl;
    std::cout<<"--------before 2int8--------"<<std::endl;
    showDevice(this->blobs_int8_[0].get()->gpu_data(),20);
#endif
    weight2int8(weightFp32.count(),weightFp32.gpu_data(),this->blobs_int8_[0].get()->mutable_gpu_data(),this_t*-1.0,this_t,weight_temp_unit_sacle_1,0,false);
#ifdef SHOW_WEIGHT2INT8
    std::cout<<"--------after 2int8--------"<<std::endl;
    showDevice(this->blobs_int8_[0].get()->gpu_data(),20);
    std::cout<<"--------weightFp32--------"<<std::endl;
    showDevice(weightFp32.gpu_data(),20);
#endif
  }
  LOG(INFO)<<"weight_temp_unit_sacle = "<<this->weight_temp_unit_sacle<<";  weight_temp_unit_sacle_1 = "<<this->weight_temp_unit_sacle_1;
}
template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::computeInt8input(int *idx, const Dtype t1,const Dtype t2)
{
  Dtype t1_pos = t1>0?t1:(-t1);
  //Dtype t2_pos = t2>0?t2:(-t2);
  Dtype each_seg = (t2-t1);
  int this_idx = (*idx-this->preTestBatches);
  int whichInputSeg=this_idx/(weight_adjust_segment_count*weight_adjust_each_count)/input_adjust_each_count;
  Dtype this_t = t1_pos+each_seg*whichInputSeg/input_adjust_segment_count;
  this->blobs_[0].get()->mutable_cpu_data()[2]=this_t*-1;
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
  {weight_uni_scale=254.0/(maxT-minT);}
  if(!doBias)
  {
    CHECK(minT < maxT);
    caffe_gpu_quantize_nobias(count, fp32weights, int8weight, minT, maxT, weight_uni_scale);//这里是希望做乘法
  }
  else
  {
    LOG(FATAL)<<"NOT IMPELMENT!";
  }
}


template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const signed char* weights, Dtype* output, bool skip_im2col) {
#ifdef SHOW_INPUT2INT8

    std::cout<<"-------------input----------------"<<std::endl; 
    showDevice(input,50); 
    std::cout<<"----------------------before col_buffer_----------------------"<<std::endl; 
    showDevice(col_buffer_.cpu_data(),50); 
#endif
#ifdef SHOW_FP32COL    
   if (!is_1x1_) 
   {
     std::cout<<"-------------input SHOW_INPUT2INT8-------------"<<std::endl; 
          showDevice(input,50); 
          im2col_gpu(input, conv_in_channels_,
                      conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                      kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                      pad_.cpu_data()[0], pad_.cpu_data()[1],
                      stride_.cpu_data()[0], stride_.cpu_data()[1],
                      dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buffer_show_.mutable_gpu_data());

        std::cout<<"--------------fp32 col_buffer_show_---------------"<<std::endl; 
        showDevice(col_buffer_show_.gpu_data(),500); 
   }
#endif
  const signed char* col_buff;
  if (!is_1x1_) {

    
          CHECK( num_spatial_axes_ == 2);
          im2col_gpu_quantized(input, conv_in_channels_, conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                                                    kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                                                    pad_.cpu_data()[0], pad_.cpu_data()[1],
                                                    stride_.cpu_data()[0], stride_.cpu_data()[1],
                                                    dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buffer_.mutable_gpu_data(),this->blobs_[0].get()->cpu_data()[2],this->blobs_[0].get()->cpu_data()[3],this->input_temp_unit_sacle_1);
    col_buff = col_buffer_.gpu_data();
#ifdef SHOW_INPUT2INT8
          std::cout<<"--------------input_T1 = "<<this->blobs_[0].get()->cpu_data()[2]<<"-------------"<<std::endl; 
          std::cout<<"--------------input_T2 = "<<this->blobs_[0].get()->cpu_data()[3]<<"-------------"<<std::endl; 
          std::cout<<"--------------input_sacle = "<<this->input_temp_unit_sacle_1<<"-------------"<<std::endl; 
          std::cout<<"-------------after col_buffer_-------------"<<std::endl; 
          showDevice(col_buffer_.gpu_data(),500);
          
#endif
  }
  else
  {
        im2col_1x1_gpu_quantized(col_buffer_.count(), input, col_buffer_.mutable_gpu_data(), this->blobs_[0].get()->mutable_cpu_data()[2],this->blobs_[0].get()->mutable_cpu_data()[3],this->input_temp_unit_sacle);
        col_buff = col_buffer_.gpu_data();
  }
  	int newK=0;
	int newN=0;
	signed char *d_A_new, *d_B_new;
	int * d_C_32_new;
	int m = conv_out_channels_ / group_;
	int needReshape = getNewDim(conv_out_spatial_dim_,kernel_dim_,&newN,&newK);
	int bigger_count1=m*newK;
	int bigger_count2=newK*newN;
	int bigger_count=m*newN;
	LOG(INFO)<<"needReshape = "<<needReshape<<" AND (newN,newK)=( "<<newN<<", "<<newK<<" ) THE OLD = ("<<conv_out_spatial_dim_<<", "<<kernel_dim_<<" )";
	if(needReshape>0) 
	{
		(cudaMalloc(&d_A_new, m * newK * sizeof(signed char)));
		(cudaMalloc(&d_B_new, newK * newN * sizeof(signed char)));
		(cudaMalloc(&d_C_32_new, m * newN * sizeof(signed char)));
	}
	
  for (int g = 0; g < group_; ++g) 
  {
	if(needReshape>0) 
	{  
	  	_copy_Data<<<CAFFE_GET_BLOCKS(bigger_count1), CAFFE_CUDA_NUM_THREADS>>>(bigger_count1,weights + weight_offset_ * g,d_A_new,m,kernel_dim_,m,newK);
		_copy_Data<<<CAFFE_GET_BLOCKS(bigger_count2), CAFFE_CUDA_NUM_THREADS>>>(bigger_count2,col_buff + col_offset_ * g,d_B_new,kernel_dim_,conv_out_spatial_dim_,newK,newN);
		caffe_gpu_iGemm(CblasNoTrans, CblasNoTrans, m, newN, newK,
        1, d_A_new, d_B_new,
        0, d_C_32_new);
		_copy_Data_back<<<CAFFE_GET_BLOCKS(bigger_count), CAFFE_CUDA_NUM_THREADS>>>(bigger_count,int32out.mutable_gpu_data() + output_offset_ * g,d_C_32_new,m,conv_out_spatial_dim_,m,newN);
	}  
	else
	{
		caffe_gpu_iGemm(CblasNoTrans, CblasNoTrans, m, newN, newK,
        1, weights + weight_offset_ * g, col_buff + col_offset_ * g,
        0, int32out.mutable_gpu_data() + output_offset_ * g);
	}
  }
  if(needReshape>0) 
  {  
	cudaFree(d_A_new);
	cudaFree(d_B_new);
	cudaFree(d_C_32_new);
  }

    
  int2Dtype(int32out.count(),int32out.gpu_data(),output,this->input_temp_unit_sacle*this->weight_temp_unit_sacle);

}
template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias)
{}


__global__ void getRelativeEntropy(int N, const float*data, const float*gtData,float*outData, float relativeLog)
{
	CUDA_KERNEL_LOOP(idx,N){
		//outData[idx] = 1.0;//gtData[idx]*(log(data[idx]/gtData[idx])-relativeLog);
		outData[idx] = gtData[idx]*(log(data[idx]/gtData[idx])-relativeLog);
	}
}
__global__ void getRelativeEntropy(int N, const double*data, const double*gtData,double*outData, double relativeLog)
{
	CUDA_KERNEL_LOOP(idx,N){
		outData[idx] = gtData[idx]*(log(data[idx]/gtData[idx])-relativeLog);
	}
}

template <typename Dtype>
Dtype ConvInt8withKLLayer<Dtype>::figureRelativeEntropy(Blob<Dtype>*int8Blob, Blob<Dtype>*gtBlob)
{
	
	CHECK(int8Blob->count() == gtBlob->count());
	CHECK(int8Blob->count() == this->relativeEntropyBlob.count());
	
	caffe_gpu_abs(int8Blob->count(), int8Blob->gpu_data(), int8Blob->mutable_gpu_data());
	caffe_gpu_abs(int8Blob->count(), gtBlob->gpu_data(), gtBlob->mutable_gpu_data());
	LOG(INFO)<<int8Blob->asum_data();//有同步问题！！！！
	Dtype int8Sum = int8Blob->asum_data();
	Dtype gtSum = gtBlob->asum_data();
	Dtype relativeLog = log(gtSum/int8Sum);
	LOG(INFO)<<int8Sum;
	//return 0;
	int _count=int8Blob->count();
	getRelativeEntropy<<<CAFFE_GET_BLOCKS(_count), CAFFE_CUDA_NUM_THREADS>>>(
      int8Blob->count(), int8Blob->gpu_data(), gtBlob->gpu_data(), this->relativeEntropyBlob.mutable_gpu_data(),relativeLog);

	Dtype entropy_=-1;
	LOG(INFO)<<int8Blob->count();
	LOG(INFO)<<this->relativeEntropyBlob.count();
	LOG(INFO)<<relativeLog;
    //caffe_gpu_asum(relativeEntropyBlob.count(), relativeEntropyBlob.gpu_data(), &(relativeEntropyBlob.mutable_gpu_diff()[0]));
	
	//showDevice(this->relativeEntropyBlob.gpu_data(),50);
	//showDevice(this->relativeEntropyBlob.gpu_data(),int8Blob->count());
	
	LOG(INFO)<<this->relativeEntropyBlob.asum_data();
	exit(0);
	//LOG(INFO)<<gtSum;
	//LOG(INFO)<<relativeLog;
	//std::cout<<entropy_<<std::endl;
	return entropy_/gtSum;
}


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
        LOG(INFO)<<"input  region : "<<this->maxAndMin.cpu_data()[1]<<"\t"<<this->maxAndMin.cpu_data()[0];
        LOG(INFO)<<"weight region : "<<this->maxAndMin.cpu_data()[3]<<"\t"<<this->maxAndMin.cpu_data()[2];
        LOG(INFO)<<preTestIdx<<" < "<<preTestBatches;
    }

    if(preTestIdx<preTestBatches) return;
    computeInt8Weight(&preTestIdx,this->maxAndMin.cpu_data()[2],this->maxAndMin.cpu_data()[3]);
    computeInt8input(&preTestIdx,this->maxAndMin.cpu_data()[0],this->maxAndMin.cpu_data()[1]);
    preTestIdx++;

  const signed char* weight = this->blobs_int8_[0]->gpu_data();
  //for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top_result.mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
		  LOG(INFO)<<"forward_gpu_gemm DONE";

      if (this->bias_term_) {
		  LOG(INFO)<<"bias_term_ BEGIN";
		  LOG(INFO)<<"this->blobs_[1]->SHAPE_STRING = "<<this->blobs_[1]->shape_string();
		  LOG(INFO)<<top_result.shape_string();
		  LOG(INFO)<<this->blobs_[1]->cpu_data()[1];
        const Dtype* bias = this->blobs_[1]->gpu_data();
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                  out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
                  (Dtype)1., top_data + n * this->top_dim_);
		LOG(INFO)<<"bias_term_ DONE";
      }
    }

    // std::cout<<"*************int8 result*************"<<std::endl; 
    // showDevice(top_result.cpu_data(),50);
    // std::cout<<"*************fp32 result*************"<<std::endl; 
    // showDevice(bottom[1]->cpu_data(),50);
    // std::cout<<"============================"<<std::endl; 
	//LOG(INFO)<<top_result.asum_data();
	//LOG(INFO)<<bottom[1]->asum_data();
	//figureRelativeEntropy(&top_result, bottom[1]);
	
	top[0]->mutable_cpu_data()[0]=figureRelativeEntropy(&top_result, bottom[1]);
}

template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(INFO)<<2;
    LOG(FATAL)<<"NOT IMPLEMENTED";
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvInt8withKLLayer);

}  // namespace caffe
