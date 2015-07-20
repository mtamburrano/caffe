#include <opencv2/core/core.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  const bool generate_data  = this->layer_param_.image_data_param().generate_data();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  bool is_first_sample_image = true;
  while (infile >> filename >> label) {
    if(is_first_sample_image) {
      sample_image_ = std::make_pair(filename, label);
      is_first_sample_image = false;
    }
    else
      lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle() && generate_data == false) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + sample_image_.first,
                                    new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);

  if(generate_data) {

    int width_rendered_image = new_width;
    int height_rendered_image = new_height;
    float radius = 0.45;
    int iter_depth = 2;
    int numLight = 5;
    int numScale = 3;
    float stepScale = 0.05;
    bool renderOnlyTopSphere = true;
    int degStep = 30;

    data_gen_ = DataGenerator( width_rendered_image, height_rendered_image, radius, iter_depth, numLight, numScale, stepScale, renderOnlyTopSphere, degStep );
    for (int l = 0; l < lines_.size(); l+=2) {
      cv::viz::Mesh m = cv::viz::Mesh::loadOBJ(lines_[l].first);
      cv::Mat textureObj = cv::imread( lines_[l+1].first );
      m.texture = textureObj;
      cv::Ptr<cv::viz::WMesh> mesh_widget = cv::Ptr<cv::viz::WMesh>( new cv::viz::WMesh( m ) );
      meshes_.push_back(mesh_widget);
    }
    //batch size deve essere divisibile per il numero di oggetti e il numero di immagini generate per ogni oggetto ad ognipose
    CHECK_EQ(batch_size%((meshes_.size())*(data_gen_.getNumRendersPerPose())), 0) <<
        "Batch size non divisibile per num_objects*data_gen_->getNumRendersPerPose()("<<meshes_.size()<<"*"<<data_gen_.getNumRendersPerPose()<<")";

  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const bool generate_data  = image_data_param.generate_data();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + sample_image_.first,
      new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();

  if(generate_data == false) {
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      // get a blob
      timer.Start();
      CHECK_GT(lines_size, lines_id_);
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = this->prefetch_data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();

      prefetch_label[item_id] = lines_[lines_id_].second;
      // go to the next iter
      lines_id_++;
      if (lines_id_ >= lines_size) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        lines_id_ = 0;
        if (this->layer_param_.image_data_param().shuffle()) {
          ShuffleImages();
        }
      }
    }
  }
  else
  {
    const int objects_size = meshes_.size();
    //divido il batch size per il numero di oggetti e il numero di immagini generate per ogni oggetto ad ognipose
    int num_items_per_object = batch_size / (objects_size*data_gen_.getNumRendersPerPose());
    int itemid = 0;
    for (int ni = 0; ni < num_items_per_object; ++ni) {

      //object_id viene incrementato dentro il for per prendere ogni coppia
      int index_line = 0;
      for (int object_id = 0; object_id < objects_size; ++object_id) {

        cv::Ptr<cv::viz::WMesh> object = meshes_[object_id];
        int label = lines_[object_id*2].second;
        data_gen_.loadObject(object);
        data_gen_.generateAndSetBackgrounds(PERLIN);
        vector<cv::Mat> rendered_images;
        data_gen_.getRenderedScene(rendered_images);
        for (int ri = 0; ri < rendered_images.size(); ++ri) {
          // get a blob
          timer.Start();
          cv::Mat cv_img = rendered_images[ri];
          if(is_color == false && cv_img.channels() > 1){
            cv::cvtColor(cv_img, cv_img, CV_BGR2GRAY);
          }
          ///DEBUG
          /*
          imshow("generated",cv_img);
          LOG(INFO) << "label: "<< label<<endl;
          waitKey();
          */
          ////////


          CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
          read_time += timer.MicroSeconds();
          timer.Start();
          // Apply transformations (mirror, crop...) to the image
          int offset = this->prefetch_data_.offset(itemid);
          this->transformed_data_.set_cpu_data(prefetch_data + offset);
          this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
          trans_time += timer.MicroSeconds();

          prefetch_label[itemid] = label;
          // go to the next iter
          ++itemid;
        }
        data_gen_.removeObject();
      }

      data_gen_.nextPose();
    }


  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
