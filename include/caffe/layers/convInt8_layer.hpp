#ifndef CAFFE_CONVINT8_LAYER_HPP_
#define CAFFE_CONVINT8_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template <typename Dtype>
class ConvInt8Layer : public Layer<Dtype> {
 public:
  explicit ConvInt8Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ConvInt8"; }
//如果是bottom=1，那么就会把它当成fp32，进行量化（除非设置inputNeedReQuantize）
//如果是bottom=2，那么就会把第一个当作int8，第二个当作scale
//如果是top=1，那么就会把它当成fp32
//如果是top=2，那么就会把第一个当作int8，第二个当作scale
//一般情况bottom和top都是size=1
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int MaxNumBottomBlobs() const { return 2; }
  virtual inline int MinNumTopBlobs() const { return 1; }
  virtual inline int MaxNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int conv_learnable_blob_size;
  
    shared_ptr<Layer<Dtype> > internalConv_layer_;//这里主要是为了避免计算output而已
    bool inputNeedReQuantize;//对input来说，是不需要requantize，一般的convInt8，接受输入是fp32，然后再该层中进行量化编码，因此这一层里面需要有个量化范围，也应该作为一个blobs，但是不学习
    bool hasInputBias;
    bool hasWeightBias;
  std::vector<Blob<Dtype>*> internalConv_bottomVect;//for internalConv_layer_
  std::vector<Blob<Dtype>*> internalConv_topVect;//for internalConv_layer_
  
  bool weight32convert2int8;
  
  void weight2int8(const int count, const Dtype*fp32weights, signed char*int8weight, const Dtype minT, const Dtype maxT, const Dtype unit_scale=-1.0, const Dtype bias=0.0, bool doBias=false);
};

}  // namespace caffe

#endif  // CAFFE_CONVINT8_LAYER_HPP_
