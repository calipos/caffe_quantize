#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/convInt8_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {



template <typename Dtype>
void ConvInt8Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConvInt8Parameter conv_int8_param = this->layer_param_.conv_int8_param();
  const ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(bottom.size()==1)<<"bottom.size()>1 not implemented yet and will be ignored in a short time.";
  CHECK(top.size()==1)<<"top.size()>1 not implemented yet and will be ignored in a short time.";
  inputNeedReQuantize = conv_int8_param.input_need_re_quantize();
  hasInputBias = conv_int8_param.has_input_bias();
  hasWeightBias = conv_int8_param.has_weight_bias();

  LayerParameter layer_param(this->layer_param_);
  layer_param.set_name(this->layer_param_.name() + "_internalConv");
  layer_param.set_type("Convolution");
  internalConv_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
  internalConv_bottomVect.clear();
  internalConv_topVect.clear();
  internalConv_bottomVect.push_back(bottom[0]);
  internalConv_topVect.push_back(top[0]);
  internalConv_layer_->LayerSetUp(internalConv_bottomVect,internalConv_topVect);
  
  this->conv_learnable_blob_size=this->layer_param_.convolution_param().bias_term()==true?9:8;
  /*
  // blobs_[0]:weight_32
  // blobs_[1]:weight_8
  // blobs_[2]:input_t1
  // blobs_[3]:input_t2
  // blobs_[4]:weight_t1
  // blobs_[5]:weight_t2
  // blobs_[6]:input_bias
  // blobs_[7]:weight_bias
  // blobs_[8]:bias*/
  this->blobs_.resize(this->conv_learnable_blob_size);
  this->blobs_int8_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>(this->internalConv_layer_->blobs()[0]->shape()));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
  
  this->blobs_[1].reset(new Blob<Dtype>(std::vector<int>{1}));//fake 真正的int8存在this-〉blobs_int8_
  this->blobs_int8_[0].reset(new Blob<signed char>(this->internalConv_layer_->blobs()[0]->shape()));
  this->blobs_[2].reset(new Blob<Dtype>(std::vector<int>{1}));
  this->blobs_[3].reset(new Blob<Dtype>(std::vector<int>{1}));
  this->blobs_[4].reset(new Blob<Dtype>(std::vector<int>{1}));
  this->blobs_[5].reset(new Blob<Dtype>(std::vector<int>{1}));
  this->blobs_[6].reset(new Blob<Dtype>(std::vector<int>{1}));
  this->blobs_[7].reset(new Blob<Dtype>(std::vector<int>{1}));
  
  this->blobs_[2].get()->mutable_cpu_data()[0]=-1;
  this->blobs_[3].get()->mutable_cpu_data()[0]=-1;
  this->blobs_[4].get()->mutable_cpu_data()[0]=-1;
  this->blobs_[5].get()->mutable_cpu_data()[0]=-1;
  this->blobs_[6].get()->mutable_cpu_data()[0]=-1;
  this->blobs_[7].get()->mutable_cpu_data()[0]=-1;
  
  if (conv_learnable_blob_size==9) 
  {
    this->blobs_[8].reset(new Blob<Dtype>(this->internalConv_layer_->blobs()[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().bias_filler()));
    bias_filler->Fill(this->blobs_[8].get());
  }
  weight32convert2int8=false;
}

template <typename Dtype>
void ConvInt8Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    internalConv_layer_->Reshape(internalConv_bottomVect,internalConv_topVect);
    top[0]->Reshape(internalConv_topVect[0]->shape());

    if(!weight32convert2int8)
    {
      
    }
    
    


}


#ifdef CPU_ONLY
STUB_GPU(ConvInt8Layer);
#endif

INSTANTIATE_CLASS(ConvInt8Layer);
REGISTER_LAYER_CLASS(ConvInt8);
}  // namespace caffe
