#ifndef CAFFE_MEMORY_DATA_LAYER_HPP_
#define CAFFE_MEMORY_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemoryDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  class MatGenerator {
   public:
    virtual void generate(int batch_size, std::vector<cv::Mat> * mats,
                          std::vector<std::vector < int > > * labels) = 0;
    virtual ~MatGenerator() {}
  };

  class DatumGenerator {
   public:
    virtual void generate(int batch_size, std::vector<Datum> * data) = 0;
    virtual ~DatumGenerator() {}
  };

  class RawPointerGenerator {
   public:
    virtual void generate(int batch_size, Dtype ** data,
                          Dtype ** labels, int * n) = 0;
    virtual ~RawPointerGenerator() {}
  };

  void inline SetMatGenerator(boost::shared_ptr<MatGenerator> callback) {
    ResetGenerators();
    generate_cv_mat_labels_cb_ = callback;
  }
  void inline SetDatumGenerator(boost::shared_ptr<DatumGenerator> callback) {
    ResetGenerators();
    generate_datum_cb_ = callback;
  }
  void inline SetRawPointerGenerator(
          boost::shared_ptr<RawPointerGenerator> callback) {
    ResetGenerators();
    generate_raw_pointer_cb_ = callback;
  }

  virtual inline const char* type() const { return "MemoryData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);
#ifdef USE_OPENCV
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
      const vector<int>& labels);
  virtual void AddMatVectorMultilabel(const vector<cv::Mat>& mat_vector,
      const vector< vector < int > >& labels);
#endif  // USE_OPENCV

  /**
   * @brief **Warning**: Reset does not perform transformations.
   *   Use `AddDatumVector` or `AddMatVector` instead if you want your input transformed.
   */
  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int n);
  void set_batch_size(int new_size);

  int batch_size() { return batch_size_; }
  int channels() { return channels_; }
  int height() { return height_; }
  int width() { return width_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void HandleGenerators();
  void inline ResetGenerators() {
    generate_cv_mat_labels_cb_ = boost::shared_ptr<MatGenerator>();
    generate_datum_cb_ = boost::shared_ptr<DatumGenerator>();
    generate_raw_pointer_cb_ = boost::shared_ptr<RawPointerGenerator>();
  }
  int batch_size_, channels_, height_, width_, data_size_, label_size_, num_labels_;
  Dtype* data_;
  Dtype* labels_;
  int n_;
  size_t pos_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
  boost::shared_ptr<MatGenerator> generate_cv_mat_labels_cb_;
  boost::shared_ptr<DatumGenerator> generate_datum_cb_;
  boost::shared_ptr<RawPointerGenerator> generate_raw_pointer_cb_;
};

}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
