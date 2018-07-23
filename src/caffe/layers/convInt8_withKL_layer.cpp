#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/convInt8_withKL_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"
namespace caffe {
template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::getFp32Weight()
{
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(this->weight_model_path, &param);
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    if(source_layer_name.compare(this->layer_fp32_name)==0)
    {
      const bool kReshape = false;
      LOG(INFO) << "Copying source layer " << source_layer_name<< ". source_layer.blobs_size()= "<<source_layer.blobs_size();
      Blob<Dtype> & target_blob = this->weightFp32;
      CHECK(source_layer.blobs_size()==conv_learnable_blob_size && target_blob.ShapeEquals(source_layer.blobs(0)));
      target_blob.FromProto(source_layer.blobs(0), kReshape);
      if(conv_learnable_blob_size==2)
      {
        if(!this->blobs_[1]->ShapeEquals(source_layer.blobs(1)))
        {
          Blob<Dtype> source_blob;
          source_blob.FromProto(source_layer.blobs(1), true);
          CHECK(this->blobs_[1]->count()==source_blob.count());
          this->blobs_[1]->FromProto(source_layer.blobs(1), true);
        }
        else
        {
          CHECK(this->blobs_[1]->ShapeEquals(source_layer.blobs(1)))<<"this->blobs_[1]="<<this->blobs_[1]->shape_string()<<"   and source_layer.blobs(1)= "<<source_layer.blobs(1).num()<<" "<<source_layer.blobs(1).channels()<<" "<<source_layer.blobs(1).height()<<" "<<source_layer.blobs(1).width()<<" ";
          this->blobs_[1]->FromProto(source_layer.blobs(1), false);
        }
      }      
      weightFp32HasExtracted = true;
      break;
    }
    else
    {
      continue;
    }
    if(!weightFp32HasExtracted) LOG(FATAL) <<"Extract weights has failed";
  }
}

template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) 
  {
    CHECK_EQ(num_spatial_axes_, 2) << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())<< "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else 
  {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)<< "kernel_size must be specified once, or once per spatial dimension "<< "(kernel_size specified " << num_kernel_dims << " times; "<< num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) 
      {
        kernel_shape_data[i] = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) 
  {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) 
  {
    CHECK_EQ(num_spatial_axes_, 2)<< "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())<< "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else 
  {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)<< "stride must be specified once, or once per spatial dimension "<< "(stride specified " << num_stride_dims << " times; "<< num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) 
    {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride : conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) 
  {
    CHECK_EQ(num_spatial_axes_, 2)<< "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())<< "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else 
  {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 || num_pad_dims == num_spatial_axes_)<< "pad must be specified once, or once per spatial dimension "<< "(pad specified " << num_pad_dims << " times; "<< num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) 
    {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad : conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 || num_dilation_dims == num_spatial_axes_) << "dilation must be specified once, or once per spatial dimension "<< "(dilation specified " << num_dilation_dims << " times; "<< num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) 
  {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation : conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) 
  {
    is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0) << "Number of output should be multiples of group.";

  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;

  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) 
  {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  
  
  vector<int> bias_shape({bias_term_, num_output_});

/***************************************************************/
    const ConvInt8KLParameter conv_int8_withkl_param = this->layer_param_.conv_int8_withkl_param();
    bottomFp32Idx = conv_int8_withkl_param.bottom_fp32_idx();
    preTestBatches = conv_int8_withkl_param.pre_test_batches();
    // input_min = -1e5;
    // input_max = -1e5;
    // weight_min = -1e5;
    // weight_max = -1e5;
    maxAndMin.Reshape(std::vector<int>({1,4}));
    maxAndMin.mutable_cpu_data()[0]=-1e7;
    maxAndMin.mutable_cpu_data()[1]=-1e7;
    maxAndMin.mutable_cpu_data()[2]=-1e7;
    maxAndMin.mutable_cpu_data()[3]=-1e7;
    input_scale_t1=0;
    input_scale_t2=0;
    isFirstGetMaxMin=false;
    weight_temp_unit_sacle=0;
    weight_temp_unit_sacle_1=0;
    input_temp_unit_sacle=0;
    input_temp_unit_sacle_1=0;
    current_weight_adjust_segment_idx=-1;
    input_adjust_segment_count = conv_int8_withkl_param.input_adjust_segment_count();
    input_adjust_each_count = conv_int8_withkl_param.input_adjust_each_count();
    weight_adjust_segment_count = conv_int8_withkl_param.weight_adjust_segment_count();
    weight_adjust_each_count = conv_int8_withkl_param.weight_adjust_each_count();
    weight_model_path = conv_int8_withkl_param.weight_model_path();
    layer_fp32_name = conv_int8_withkl_param.layer_fp32_name();
    LOG(INFO)<<" **  the weights of "<<layer_fp32_name<<" will be extracted from "<<weight_model_path;
    weightAlready = false;
    CHECK(preTestBatches>0)<<"test for getting the region of input";
    CHECK(input_adjust_segment_count>0);
    CHECK(input_adjust_each_count>0);
    CHECK(weight_adjust_segment_count>0);
    CHECK(weight_adjust_each_count>0);
    preTestIdx=0;


    const ConvInt8Parameter conv_int8_param = this->layer_param_.conv_int8_param();
  //CHECK(bottom.size()==2)<<"bottom.size()>1 not implemented yet and will be ignored in a short time.";
  CHECK(top.size()==1)<<"top.size()>1 not implemented yet and will be ignored in a short time.";
  inputNeedReQuantize = conv_int8_param.input_need_re_quantize();
  hasInputBias = conv_int8_param.has_input_bias();
  hasWeightBias = conv_int8_param.has_weight_bias();
  this->conv_learnable_blob_size=this->layer_param_.convolution_param().bias_term()==true?2:1;
  /*
  // blobs_[0]:weight_32  不存储fp32的权，大小也只有一个单位     
     blobs_[1]:       weight_8：weight_32
          //不存储fp32的权，大小也只有一个单位，正数表示blobs_[1]:weight_8已经久违，负数则表示blobs_[1]:weight_8没有准备好，blobs_[1]:weight_8的准备过程发生在int8的kl阶段，在彼阶段，读进fp32，并按相关参数转为int8，再存入int8，最后保存在caffemodel，也就是说每一次读取caffemodel，这里必然是准备好了的int8weight
  // blobs_[2]:input_t1
  // blobs_[3]:input_t2
  // blobs_[4]:weight_t1
  // blobs_[5]:weight_t2
  // blobs_[6]:input_bias
  // blobs_[7]:weight_bias
  // blobs_[8]:bias*/
  this->blobs_.resize(this->conv_learnable_blob_size);
  this->blobs_int8_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>(std::vector<int>{1,8}));  
  for(int i=0;i<8;i++) this->blobs_[0].get()->mutable_cpu_data()[i]=-1;
  this->blobs_int8_[0].reset(new Blob<signed char>(weight_shape));
  weightFp32.Reshape(weight_shape);
  weightFp32HasExtracted = false;
  for(int i=0;i<this->blobs_int8_[0].get()->count();i++) 
      this->blobs_int8_[0].get()->mutable_cpu_data()[i]=i-10;
  if (conv_learnable_blob_size==2) 
  {
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
  kernel_dim_ = this->blobs_int8_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
}

template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
        const int first_spatial_axis = channel_axis_ + 1;
        CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)<< "bottom num_axes may not change.";
        num_ = bottom[0]->count(0, channel_axis_);
        CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)<< "Input size incompatible with convolution kernel.";

        bottom_shape_ = &bottom[0]->shape();
        compute_output_shape();
        vector<int> top_shape(bottom[0]->shape().begin(),
            bottom[0]->shape().begin() + channel_axis_);
        top_shape.push_back(num_output_);
        for (int i = 0; i < num_spatial_axes_; ++i) 
        {
          top_shape.push_back(output_shape_[i]);
        }
        top[0]->Reshape(std::vector<int>({1}));
        top_result.Reshape(top_shape);
        CHECK(top_shape.size()==4)<<"only support 4dim!";

          int32out.Reshape(top_shape);
        
        conv_out_spatial_dim_ = top_result.count(first_spatial_axis);
        

        
        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
        // Setup input dimensions (conv_input_shape_).
        vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
        conv_input_shape_.Reshape(bottom_dim_blob_shape);
        int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
        for (int i = 0; i < num_spatial_axes_ + 1; ++i) 
        {
          conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
        }
        // The im2col result buffer will only hold one image at a time to avoid
        // overly large memory usage. In the special case of 1x1 convolution
        // it goes lazily unused to save memory.
        col_buffer_shape_.clear();
        col_buffer_shape_.push_back(kernel_dim_ * group_);
        for (int i = 0; i < num_spatial_axes_; ++i) 
        {
          col_buffer_shape_.push_back(output_shape_[i]);
        }
        col_buffer_.Reshape(col_buffer_shape_);
        //if(is_1x1_) CHECK(col_buffer_.count()==bottom[0]->count())<<col_buffer_.count()<<" : "<<bottom[0]->count();
#ifdef SHOW_FP32COL
        col_buffer_show_.Reshape(col_buffer_shape_);
#endif
        //************************************
        bottom_dim_ = bottom[0]->count(channel_axis_);
        top_dim_ = top_result.count(channel_axis_);
        num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
        num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
        // Set up the all ones "bias multiplier" for adding biases by BLAS
        out_spatial_dim_ = top_result.count(first_spatial_axis);
        if (bias_term_) {
          vector<int> bias_multiplier_shape(1, out_spatial_dim_);
          bias_multiplier_.Reshape(bias_multiplier_shape);
          caffe_set(bias_multiplier_.count(), Dtype(1),
              bias_multiplier_.mutable_cpu_data());
        }
        CHECK(conv_out_spatial_dim_%4==0)<<conv_out_spatial_dim_<<" - "<<top_result.shape_string();
        /*******************************************************************/
        inputInt8.Reshape(col_buffer_shape_);
        LOG(INFO)<<"caffe_gpu_iGemm PARAM : "<<conv_out_channels_ / group_<<" "<<conv_out_spatial_dim_<<" "<<kernel_dim_;

}

template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LOG(FATAL)<<"NOT IMPLEMENTED";
}

template <typename Dtype>
void ConvInt8withKLLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    LOG(FATAL)<<"NOT IMPLEMENTED";
}


#ifdef CPU_ONLY
STUB_GPU(ConvInt8withKLLayer);
#endif

INSTANTIATE_CLASS(ConvInt8withKLLayer);
REGISTER_LAYER_CLASS(ConvInt8withKL);
}  // namespace caffe
