#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), bottom.size()-1) <<
      "Top.size() should be equal to bottom.size() - 1";
  const FilterParameter& filter_param = this->layer_param_.filter_param();
  first_reshape_ = true;
  need_back_prop_.clear();
  std::copy(filter_param.need_back_prop().begin(),
      filter_param.need_back_prop().end(),
      std::back_inserter(need_back_prop_));
  CHECK_NE(0, need_back_prop_.size()) <<
      "need_back_prop param needs to be specified";
  CHECK_EQ(top.size(), need_back_prop_.size()) <<
      "need_back_prop.size() needs to be equal to top.size()";
}

template <typename Dtype>
void FilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0] is the "SELECTOR_blob"
  // bottom[1+] are the blobs to filter
  int num_items = bottom[0]->num();
  
  CHECK_EQ(bottom[0]->channels(), 1) <<
        "Selector blob (bottom[0]) must have channels == 1";
  CHECK_EQ(bottom[0]->width(), 1) <<
        "Selector blob (bottom[0]) must have width == 1";
  CHECK_EQ(bottom[0]->height(), 1) <<
        "Selector blob (bottom[0]) must have height == 1";
  for(int i = 1; i < bottom.size(); i++) {
    CHECK_EQ(num_items, bottom[i]->num()) <<
        "Each bottom should have the same dimension as bottom[0]";
  }

  const Dtype* bottom_data_SELECTOR = bottom[0]->cpu_data();
  indices_to_forward_.clear();

  // look for non-zero elements in bottom[0]. Items of each bottom that
  // have the same index as the items in bottom[0] with value == non-zero
  // will be forwarded
  for (size_t item_id = 0; item_id < num_items; ++item_id) {
    //we don't need an offset because item size == 1
    const Dtype* tmp_data_SELECTOR = bottom_data_SELECTOR + item_id;
    if(*tmp_data_SELECTOR != 0) {
      indices_to_forward_.push_back(item_id);
    }
  }

  // only filtered items will be forwarded
  int new_tops_num = indices_to_forward_.size();
  // init
  if (first_reshape_) {
    new_tops_num = bottom[1]->num();
    first_reshape_ = false;
  }

  for(size_t t = 0; t < top.size(); t++) {
    top[t]->Reshape(new_tops_num,
        bottom[t+1]->channels(),
        bottom[t+1]->height(),
        bottom[t+1]->width());
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[0])
  for(size_t b = 1; b < bottom.size(); b++) {
    const Dtype* bottom_data = bottom[b]->cpu_data();
    Dtype* top_data = top[b-1]->mutable_cpu_data();
    size_t dim = bottom[b]->count()/bottom[b]->num();

    for (size_t n = 0; n < new_tops_num; n++) {
      int offset = indices_to_forward_[n];
      int data_offset_top = dim*n;
      int data_offset_bottom = dim*offset;

      caffe_copy(dim, bottom_data+data_offset_bottom,
          top_data+data_offset_top);
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for(size_t i = 1; i < propagate_down.size(); i++) {
    // bottom[0] is the selector and never needs backpropagation
    // so we can start from i = 1 and index each top with i-1
    if (propagate_down[i] && need_back_prop_[i-1]) {
      const int dim = top[i-1]->count()/top[i-1]->num();
      int index_top = 0;
      std::vector<double> zeros(dim, 0.0);
      for (size_t n = 0; n < bottom[i]->num(); n++) {
        int offset = indices_to_forward_[n];
        int data_offset_bottom = dim*n;
        if (n != offset) {  // this data was not been forwarded
          caffe_copy(dim, reinterpret_cast<Dtype*>(&zeros[0]),
              bottom[i]->mutable_cpu_diff() + data_offset_bottom);
        } else {  // this data was been forwarded
          int data_offset_top = dim*index_top;
          index_top++;
          caffe_copy(dim, top[i-1]->mutable_cpu_diff() + data_offset_top,
              bottom[i]->mutable_cpu_diff() + data_offset_bottom);
        }
      }
    }
  }
    
}

#ifdef CPU_ONLY
STUB_GPU(FilterLayer);
#endif

INSTANTIATE_CLASS(FilterLayer);
REGISTER_LAYER_CLASS(FILTER, FilterLayer);
}  // namespace caffe
