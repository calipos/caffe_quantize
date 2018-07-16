#include <boost/thread.hpp>
#include <iostream>
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}
template <typename Dtype>
void Layer<Dtype>::showInt8blobs(int i, int count)
{
  std::cout<<blobs_int8_[i]->shape_string()<<std::endl;
  for(int j=0;j<count*5;j+=5)
  {
    std::cout<<(int)(blobs_int8_[i]->cpu_data()[j])<<"\t"
            <<(int)(blobs_int8_[i]->cpu_data()[j+1])<<"\t"
            <<(int)(blobs_int8_[i]->cpu_data()[j+2])<<"\t"
            <<(int)(blobs_int8_[i]->cpu_data()[j+3])<<"\t"
            <<(int)(blobs_int8_[i]->cpu_data()[j+4])<<std::endl;
  }
}


INSTANTIATE_CLASS(Layer);




}  // namespace caffe
