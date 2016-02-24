#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_euclidean_layer.hpp"
#include "caffe/util/math_functions.hpp"

double distanceEuclidean(double x1, double y1, double x2, double y2)
{
    double x = x1 - x2;
    double y = y1 - y2;
    double dist;

    dist = pow(x,2)+pow(y,2);
    dist = sqrt(dist);

    return dist;
}

namespace caffe {

template <typename Dtype>
void AccuracyEuclideanLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
  dist_threshold_ = this->layer_param_.accuracy_euclidean_param().distance_threshold();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyEuclideanLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);

  CHECK_EQ(inner_num_, 2)
      << "PER ORA L'ACCURACY_EUCLIDEAN SOPPORTA SOLO UNA COPPIA DI COORDINATE (X,Y)";
  CHECK_EQ(inner_num_, 2)
      << "PER ORA L'ACCURACY_EUCLIDEAN SOPPORTA SOLO UNA COPPIA DI COORDINATE (X,Y)";
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top[1]->Reshape(top_shape);
  }
  /*if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }*/
}

template <typename Dtype>
void AccuracyEuclideanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  /*if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }*/

  int count = 0;
  for (int i = 0; i < outer_num_; ++i)
  {
    int x_p, x_l, y_p, y_l;
    //l'accuracy per questo item viene inizializzatad 1, viene posta a 0 se
    //  se viene sbagliata la predict di una qualsiasi delle classi
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      //if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;

      if(j == 0) //coordinata x
      {
        x_p = bottom_data[i * inner_num_ + j];
        x_l = label_value;
      }
      else if(j == 1) //coordinata y
      {
        y_p = bottom_data[i * inner_num_ + j];
        y_l = label_value;
      }
      //std::cout << "bottom: "<<bottom_data[i * inner_num_ + j]<<std::endl;
      //std::cout << "label_value: "<<label_value<<std::endl;

      //////////TOP_K != 1 NON FUNZIONA
    }
    double dist = distanceEuclidean(x_p, y_p, x_l, y_l);
    if (dist <= dist_threshold_) {
      accuracy++;
    }

  }

  top[0]->mutable_cpu_data()[0] = accuracy / outer_num_;
  //if (top.size() > 1) {
    //top[1]->mutable_cpu_data()[0] = multi_task_accuracy / outer_num_;
  ////////////MULTI TASK ACCURACY NON SUPPORTATA PER ORA PER EUCLIDEAN ACCURACY/////////////
}

INSTANTIATE_CLASS(AccuracyEuclideanLayer);
REGISTER_LAYER_CLASS(AccuracyEuclidean);

}  // namespace caffe
