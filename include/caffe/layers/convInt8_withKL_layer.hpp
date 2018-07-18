#ifndef CAFFE_CONVINT8_WITHKL_LAYER_HPP_
#define CAFFE_CONVINT8_WITHKL_LAYER_HPP_

#include <vector>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_conv_layer.hpp"

//#define SHOW_FP32_OUT


namespace caffe {

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template <typename Dtype>
class ConvInt8withKLLayer : public Layer<Dtype> {
 public:
  explicit ConvInt8withKLLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
      
      
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ConvInt8withKL"; }
//如果是bottom=1，那么就会把它当成fp32，进行量化（除非设置inputNeedReQuantize）
//如果是bottom=2，那么就会把第一个当作int8，第二个当作scale
//如果是top=1，那么就会把它当成fp32
//如果是top=2，那么就会把第一个当作int8，第二个当作scale
//一般情况bottom和top都是size=1
  virtual inline int MinNumBottomBlobs() const { return 2; }
  virtual inline int MaxNumBottomBlobs() const { return 3; }
  virtual inline int MinNumTopBlobs() const { return 1; }
  virtual inline int MaxNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int conv_learnable_blob_size;  
//这里主要是为了避免计算output而已
    bool inputNeedReQuantize;//对input来说，是不需要requantize，一般的convInt8，接受输入是fp32，然后再该层中进行量化编码，因此这一层里面需要有个量化范围，也应该作为一个blobs，但是不学习
    bool hasInputBias;
    bool hasWeightBias;

  void weight2int8(const int count, const Dtype*fp32weights, signed char*int8weight, const Dtype minT, const Dtype maxT, const Dtype unit_scale=-1.0, const Dtype bias=0.0, bool doBias=false);

  Blob<signed char> inputInt8;
  Blob<Dtype> weightFp32;
  bool weightFp32HasExtracted;
  Blob<int> int32out;
  Dtype input_unit_scale;
  Dtype input_unit_scale_1;//倒数
  
  int bottomFp32Idx;
  int preTestBatches;
  // 1 :Dtype input_min;
  // 2 :Dtype input_max;
  // 3 :Dtype weight_min;
  // 4 :Dtype weight_max;
  Blob<Dtype> maxAndMin;
  Dtype input_scale_t1;
  Dtype input_scale_t2;
  bool isFirstGetMaxMin;
  int input_adjust_segment_count;
  int input_adjust_each_count;
  int weight_adjust_segment_count;
  int weight_adjust_each_count;
  int current_weight_adjust_segment_idx;
  Dtype input_best_T;
  Dtype weight_best_T;
  Dtype min_entropy;
  std::string weight_model_path;
  std::string layer_fp32_name;
  Dtype weight_temp_unit_sacle;
  Dtype weight_temp_unit_sacle_1;
  Dtype input_temp_unit_sacle;
  Dtype input_temp_unit_sacle_1;
  void getFp32Weight();
  void computeInt8Weight(int*idx,const Dtype t1,const Dtype t2);
  void computeInt8input(int*idx,const Dtype t1,const Dtype t2);
  // void computeInt8Weight(const Dtype t);
  // void computeInt8int(const Dtype t);
  // void computeInt8Weight();
  // void computeInt8int();
  bool weightAlready;
  int preTestIdx;
  //==========================================//
 protected:

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* input,
    const signed char* weights, Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }

  bool reverse_dimensions(){return false;};
  void compute_output_shape();
  Blob<int> kernel_shape_;
  Blob<int> stride_;
  Blob<int> pad_;
  Blob<int> dilation_;
  Blob<int> conv_input_shape_;
  vector<int> col_buffer_shape_;
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;
  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;

 private:

#ifndef CPU_ONLY
  inline void conv_im2col_gpu_int8(const Dtype* data, Dtype* col_buff) {
    if (num_spatial_axes_ == 2) {
      showDevice(data,10);
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
      showDevice(data,10);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if ( num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<signed char> col_buffer_;
#ifdef SHOW_FP32_OUT
  Blob<Dtype> col_buffer_show_;
#endif
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_CONVINT8_LAYER_HPP_
