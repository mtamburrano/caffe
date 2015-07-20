
#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <fstream>
#include <cfloat>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

//#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/shared_ptr.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

#define THRESH_ACCEPT 0.99
float THRESH = 0.98;

//78708801639afcdfe32e4e6c64e8164ae2182e31

void softmax_casareccio(vector<float> y, vector<float>& value)
{
  //Vector y = mlp(x); // output of the neural network without softmax activation function
  double ysum = 0;
  for(int f = 0; f < y.size(); f++)
    ysum += exp((double)y[f]);



  for(int i = 0; i<value.size(); i++)
  {
    value[i] = (float)(exp((double)value[i])/ysum);
  }
}



class Prediction{
 public:
  Prediction() {};
  Prediction(float probm, float probs, float probbg) {
   prob_mis = probm;
   prob_sen = probs;
   prob_bg = probbg;
 }
  float prob_mis;
  float prob_sen;
  float prob_bg;
};

int s2i(string s) {
  stringstream ss;
  ss << s;
  int res;
  ss >> res;
  return res;
}

string i2s(int i) {
  stringstream ss;
  ss << i;
  string res;
  ss >> res;
  return res;
}


struct delete_ptr { // Helper function to ease cleanup of container
    template <typename P>
    void operator () (P p) {
        delete p;
    }
};

string printResult(int index)
{
  string result = "";
  if(index == 0){
    cout << "    Classe Scelta: MISURATORE"<< endl;
    result = "MISURATORE";
  }
  else if(index == 1){
    cout << "    Classe Scelta: SENSORE"<< endl;
    result = "SENSORE";
  }
  else {
    cout << "    Classe Scelta: BACKGROUND"<< endl;
    result = "BACKGROUND";
  }
  return result;
}


int computePrediction(Prediction pred, bool print)
{
  vector<float> probs;
  probs.push_back(pred.prob_mis);
  probs.push_back(pred.prob_sen);
  probs.push_back(pred.prob_bg);
  
  int index_max = distance(probs.begin(), max_element(probs.begin(), probs.end()));

  if(index_max == 0 && pred.prob_mis > THRESH){
    if(print) {
      cout << " prob_mis: " << pred.prob_mis << endl;
      cout << " prob_sen: " << pred.prob_sen << endl;
      cout << " prob_bg: " << pred.prob_bg << endl;
      printResult(0);
    }
    return 0;
  }
  else if(index_max == 1 && pred.prob_sen > THRESH){
    if(print) {
      cout << " prob_mis: " << pred.prob_mis << endl;
      cout << " prob_sen: " << pred.prob_sen << endl;
      cout << " prob_bg: " << pred.prob_bg << endl;
      printResult(1);
    }
    return 1;
  }
  else {
    if(print) {
      cout << " prob_mis: " << pred.prob_mis << endl;
      cout << " prob_sen: " << pred.prob_sen << endl;
      cout << " prob_bg: " << pred.prob_bg << endl;
      printResult(2);
    }
    return 2;
  }
}



void printTxtResult(string title, vector<pair <int, pair< string, Prediction> > > results) {

  cout << endl << "**************************************************************" << endl;
  cout << "**************************************************************" << endl;
  cout << "*********************   "<<title<<"   ************************" << endl;
  cout << "**************************************************************" << endl;
  cout << "**************************************************************" << endl << endl;

  vector<string> classes;
  classes.push_back("MISURATORI");
  classes.push_back("SENSORI");
  classes.push_back("BACKGROUNDS");

  for (int c = 0; c < classes.size(); ++c){
    cout << "|/|/|/|   "<<classes[c]<<"   |/|/|/|" << endl << endl;
    int count = 0;
    for (int i = 0; i < results.size(); ++i){

      pair <int, pair< string, Prediction> > res = results[i];
      int label = res.first;
      string path = res.second.first;
      Prediction pred = res.second.second;

      if(label == c) {
        count++;
        cout << endl << "  " << path << endl;
        computePrediction(pred,/*print*/ true);
      }
    }
    cout << endl << " TOTALI: " << count << endl;
    cout << "------------------------------" << endl << endl;
  }



}





void rotate(Mat& src, Mat& dst, float angle) {
    Point2f src_center(src.cols/2.0F, src.rows/2.0F);

    cv::Mat rot_matrix = getRotationMatrix2D(src_center, angle, 1.0);

    dst.create(Size(src.size().height, src.size().width), src.type());

    warpAffine(src, dst, rot_matrix, dst.size());
}

/*Predictor::~Predictor() {
  // TODO Auto-generated destructor stub
  caffe_test_net_.release();
  for_each(result_.begin(), result_.end(), delete_ptr());
  result_.clear();
  caffe_test_net_.reset();
}*/


vector<Blob<float>*> forwardBatch(Net<float>* caffe_test_net, boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer, vector<cv::Mat>& images) {

  vector<int> labels(images.size(),-2);
  memory_data_layer->AddMatVector(images, labels);
  //PREDICT
  float loss;
  return caffe_test_net->ForwardPrefilled(&loss);

}

void getPredictions(Net<float>* caffe_test_net, int num_images_forwarded, vector<Prediction>& predictions) {
  float class_predicted_mis = 99999;
  float class_predicted_sen = 99998;
  float class_predicted_bg = 99997;
  shared_ptr<Blob<float> > probs = caffe_test_net->blob_by_name("probs");
  const float* bottom_data = probs->cpu_data();

  for(int i = 0; i<num_images_forwarded; i++) {
    class_predicted_mis = bottom_data[(i*3)+i];
    class_predicted_sen = bottom_data[(i*3)+i+1];
    class_predicted_bg = bottom_data[(i*3)+i+2];
    Prediction pred(class_predicted_mis, class_predicted_sen, class_predicted_bg);
    predictions.push_back(pred);

  }

}

//vettore di predizioni, ogni elemento è una predizione di un'immagine ruotata
void getMeanPrediction(vector<Prediction>& predictions, Prediction& meanPrediction) {

  float meanMis = 0;
  float meanSen = 0;
  float meanBg = 0;
  for(int i = 0; i<predictions.size(); i++) {
   meanMis += predictions[i].prob_mis;
   meanSen += predictions[i].prob_sen;
   meanBg += predictions[i].prob_bg;
  }
  meanMis /= predictions.size();
  meanSen /= predictions.size();
  meanBg /= predictions.size();

  meanPrediction = Prediction(meanMis, meanSen, meanBg);
}

//vettore di vettori di predizioni, ogni vettore contiene i vettori di predizioni di un forward pass
void getMeanPrediction(vector < vector<Prediction> >& v_predictions, vector<Prediction>& meanPredictions) {

  //prendo la size del primo vettore per capire quante immagini ha ogni vettore
  int num_images = v_predictions[0].size();
  int num_forwards = v_predictions.size();

  for(int nm = 0; nm < num_images; ++nm) {

    float meanMis = 0;
    float meanSen = 0;
    float meanBg = 0;

    for(int nf = 0; nf<num_forwards; nf++) {

      meanMis += v_predictions[nf][nm].prob_mis;
      meanSen += v_predictions[nf][nm].prob_sen;
      meanBg += v_predictions[nf][nm].prob_bg;
    }

    meanMis /= num_forwards;
    meanSen /= num_forwards;
    meanBg /=num_forwards;
    Prediction meanPrediction = Prediction(meanMis, meanSen, meanBg);
    meanPredictions.push_back(meanPrediction);
  }

}





void prepareRotatedBatch(Mat& image, int batch_size, vector<Mat>& rotated_batch) {

  float angle_rotation = 360 / batch_size;
  Mat rotated_mat;

  for(int b = 0; b < batch_size; ++b) {
    if(b == 0) //immagine originale
      rotated_batch.push_back(image);
    else { //immagine originale ruotata
      rotate(image, rotated_mat, angle_rotation*b);
      rotated_batch.push_back(rotated_mat);
    }
  }
}


void getAllFilesFromTxt(string filePath, std::vector<std::string>& images_path, std::vector<int>& images_label) {
  std::ifstream file(filePath.c_str());
  std::string str;
  while (std::getline(file, str))
  {
    if (!str.empty())
    {
        char lastChar = *str.rbegin(); //get lastChar (label)
        string labelString(1,lastChar);
        images_label.push_back(s2i(labelString));
        string path_image = str.substr(0, str.find_last_of(labelString)-1);
        images_path.push_back(path_image);
        //cout << path_image << " "<< labelString << endl;
    }

  }
}

void contrastBrightness(Mat& image) {
  
  double alpha = 2; /**< Simple contrast control */
  int beta = 0;  /**< Simple brightness control */
  
  for( int y = 0; y < image.rows; y++ )
  { 
    for( int x = 0; x < image.cols; x++ )
    { 
      for( int c = 0; c < 3; c++ )
      {
        image.at<Vec3b>(y,x)[c] =
        saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
      }
    }
  }
}

void equalizeIntensity(Mat& inputImage)
{
  
  normalize(inputImage, inputImage, 0, 255, NORM_MINMAX);
  return;
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,COLOR_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        merge(channels,ycrcb);

        cvtColor(ycrcb,inputImage,COLOR_YCrCb2BGR);
        
    }
    else
      cout << "IMMAGINE NON EQUALIZZATA (è in bianco e nero) : inputImage.channels(): "<<inputImage.channels() << endl;
  

}

void transform(vector<Mat>& images) {
  for(int i = 0; i< images.size(); ++i){
    equalizeIntensity(images[i]);
    //contrastBrightness(images[i]);
  }
}

void convertToGray(vector<Mat>& images) {
  for(int i = 0; i< images.size(); ++i){
    cvtColor(images[i], images[i], COLOR_BGR2GRAY);
  }
}



//make a prediction from file_list.txt file
int main(int argc, char** argv) {

  if (argc != 5 && argc != 6 && argc != 7 && argc != 8) {
    LOG(ERROR) << "wrong input count, Usage: logicos_predict net_proto pretrained_model image_path [color|gray] (optional num_forwards)[default:1] (optional threshold)[default:hardcoded] (optional transformations: true|false)[default:false]" ;
    return 1;
  }

  std::string net_proto(argv[1]);
  std::string pretrained_model(argv[2]);
  std::string image_path(argv[3]);
  std::string color_type(argv[4]);
  
  if(color_type != "color" && color_type != "gray") {
    LOG(ERROR) << "wrong color type, Usage: logicos_predict net_proto pretrained_model image_path [color|gray] (optional num_forwards)[default:1] (optional threshold)[default:hardcoded] (optional transformations: true|false)[default:false]" ;
    return 1;
  }
  
  int num_forwards = 1;
  bool transf = false;
  if(argc > 5)
    num_forwards = std::atoi(argv[5]);
  if(argc > 6)
    THRESH = std::atof(argv[6]);
  if(argc > 7)
    transf = true;

  bool PREDICT_TXT = false;
  //se in input viene data una directory, leggiamo prendot tutti i path delle directory
  if(image_path.substr(image_path.find_last_of(".") + 1) == "txt")
    PREDICT_TXT = true;
  //}

  std::vector<std::string> filepaths;
  std::vector<int> labels;
  vector<Mat> images;
  Mat image;
  if(PREDICT_TXT) {
    getAllFilesFromTxt(image_path, filepaths, labels);
    for(int i = 0; i<filepaths.size(); i++) {
      
      image = imread(filepaths[i], 1);
      if (image.empty()){
        cout << "mat " <<filepaths[i]<<" vuota"<<endl;
         exit(0);
      }
      images.push_back(image);
    }
  }
  else { //PREDICT_DIRECTORY == false - predict only one image

    image = imread(image_path, 1);
    if (image.empty()){
      cout << "mat " <<image_path<<" vuota"<<endl;
       exit(0);
    }
    images.push_back(image);
  }
  
  imwrite("prima.jpg", images[0]);
  if(transf == true)
  {
    transform(images);
  }
  imwrite("dopo.jpg", images[0]);
  if(color_type == "gray")
  {
    convertToGray(images);
  }
  
  if(images[0].channels()  > 1)
  {
    cout<<"cazzo";
    exit(0);
  }
  

  //Setting CPU or GPU
  Caffe::set_mode(Caffe::GPU);
  int device_id = 0;
  Caffe::SetDevice(device_id);
  Net<float>* caffe_test_net = new Net<float>(net_proto, caffe::TEST);
  caffe_test_net->CopyTrainedLayersFrom(pretrained_model);
  boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(caffe_test_net->layer_by_name("data"));
  int batch_size = memory_data_layer->batch_size();


  //preparo tutti i batches così non devo fare il rotate per ogni forward
  vector < vector < Mat > > all_rotated_batches;
  for(int i = 0; i<images.size(); i++) {

    vector<Mat> rotatedBatch;
    prepareRotatedBatch(images[i],  batch_size, rotatedBatch);
    all_rotated_batches.push_back(rotatedBatch);
  }

  vector<vector < Prediction> > global_mean_predictions;
  vector<Prediction> mean_predictions;
  for(int i = 0; i<num_forwards; i++) {

    if(i > 0)
      mean_predictions.clear();

    for(int i = 0; i<images.size(); i++) {

      vector<Mat> rotatedBatch = all_rotated_batches[i];

      vector<Blob<float>*> results = forwardBatch(caffe_test_net, memory_data_layer, rotatedBatch);

      vector<Prediction> predictions;
      getPredictions(caffe_test_net, batch_size, predictions);

      Prediction meanPrediction;
      getMeanPrediction( predictions, meanPrediction);

      mean_predictions.push_back(meanPrediction);
    }

    global_mean_predictions.push_back(mean_predictions);
  }

  //riordino i prediction per fare le medie tra tutti i forwards
  if(num_forwards > 1) {

    mean_predictions.clear();
    getMeanPrediction( global_mean_predictions, mean_predictions);
  }


  int TP;
  int FP;
  int TN;
  int FN;
  int num_good = 0;
  int num_good_mis = 0;
  int num_good_sen = 0;
  int num_good_bg = 0;
  int num_bad = 0;
  int num_bad_mis = 0;
  int num_bad_sen = 0;
  int num_bad_bg = 0;
  // pair< LABEL, pair < path_image, prediction > >
  vector<pair <int, pair< string, Prediction> > > good;
  vector<pair <int, pair< string, Prediction> > > bad;

  if(PREDICT_TXT == false)
  {
    cout << "THRESHOLD: "<< THRESH << endl;
    cout << "ROTATIONS: "<< batch_size << endl << endl;
    computePrediction(mean_predictions[0], /*print result*/true);
  }
  else
  {
    for(int p = 0; p < mean_predictions.size(); ++p)
    {
      int pred_index = computePrediction(mean_predictions[p], /*print result*/false);

      if (pred_index == labels[p]){
        num_good++;
        if(labels[p] == 0)
          num_good_mis++;
        if(labels[p] == 1)
          num_good_sen++;
        if(labels[p] == 2)
          num_good_bg++;
        pair< string, Prediction> g_p = make_pair(filepaths[p], mean_predictions[p]);
        pair <int, pair< string, Prediction> > l_g_p =  make_pair(labels[p], g_p);
        good.push_back(l_g_p);
      }
      else
      {
        num_bad++;
        if(labels[p] == 0)
          num_bad_mis++;
        if(labels[p] == 1)
          num_bad_sen++;
        if(labels[p] == 2)
          num_bad_bg++;
        pair< string, Prediction> b_p = make_pair(filepaths[p], mean_predictions[p]);
        pair <int, pair< string, Prediction> > l_b_p =  make_pair(labels[p], b_p);
        bad.push_back(l_b_p);
      }
    }

    printTxtResult("GOOD", good);
    printTxtResult("BAD" , bad);
    
    cout << "THRESHOLD: "<< THRESH << endl;
    cout << "ROTATIONS: "<< batch_size << endl << endl;

    cout << "GOOD MISURATORI / TOTALI: "<< num_good_mis << "/" << (num_good_mis+num_bad_mis) << endl;
    cout << "GOOD SENSORI / TOTALI: " << num_good_sen << "/" << (num_good_sen+num_bad_sen) << endl;
    cout << "GOOD BACKGROUNDS / TOTALI "<< num_good_bg << "/" << (num_good_bg+num_bad_bg) << endl;
    cout << "Accuracy MISURATORI: " << num_good_mis / (float) (num_good_mis+num_bad_mis)  << endl;
    cout << "Accuracy SENSORI: " << num_good_sen / (float) (num_good_sen+num_bad_sen)  << endl;
    cout << "Accuracy BACKGROUNDS: " << num_good_bg / (float) (num_good_bg+num_bad_bg) << endl << endl;
    cout << "GOOD/TOTAL: " << num_good << "/"<< mean_predictions.size() << endl;
    cout << "Accuracy: " << num_good / (float) mean_predictions.size()  << endl;
    
    std::ofstream outfile;

    outfile.open("log.txt", std::ios_base::app);
    outfile << "#################################################### " << endl;
    outfile << "THRESHOLD: "<< THRESH << endl;
    outfile << "ROTATIONS: "<< batch_size << endl << endl;
    outfile << "GOOD MISURATORI / TOTALI: "<< num_good_mis << "/" << (num_good_mis+num_bad_mis) << endl;
    outfile << "GOOD SENSORI / TOTALI: " << num_good_sen << "/" << (num_good_sen+num_bad_sen) << endl;
    outfile << "GOOD BACKGROUNDS / TOTALI "<< num_good_bg << "/" << (num_good_bg+num_bad_bg) << endl;
    outfile << "Accuracy MISURATORI: " << num_good_mis / (float) (num_good_mis+num_bad_mis)  << endl;
    outfile << "Accuracy SENSORI: " << num_good_sen / (float) (num_good_sen+num_bad_sen)  << endl;
    outfile << "Accuracy BACKGROUNDS: " << num_good_bg / (float) (num_good_bg+num_bad_bg) << endl << endl;
    outfile << "GOOD/TOTAL: " << num_good << "/"<< mean_predictions.size() << endl;
    outfile << "Accuracy: " << num_good / (float) mean_predictions.size()  << endl;
    outfile << "------------------------------------------------------ " << endl;
    outfile.close();

  }


  /*string predicted = "";
  vector<float> probs;

  Mat image_flip_v, image_flip_o, image_rot, image_flip_rot;
  cv::flip(image,image_flip_o,1);
  cv::flip(image,image_flip_v,0);
  vector<string> predictions;
  vector<vector<float> > all_probs;

  cout << "*************IMMAGINE NORMALE************" <<endl;
  predicted = predict(caffe_test_net, memory_data_layer, image, probs);
  predictions.push_back(predicted);
  all_probs.push_back(probs);
  probs.clear();
  cout<<endl << "*************IMMAGINE FLIPPATA ORIZZONTALE************" <<endl;
  predicted = predict(caffe_test_net, memory_data_layer, image_flip_o, probs);
  predictions.push_back(predicted);
  all_probs.push_back(probs);
  probs.clear();
  cout<<endl << "*************IMMAGINE FLIPPATA VERTICALE************" <<endl;
  predicted = predict(caffe_test_net, memory_data_layer, image_flip_v, probs);
  predictions.push_back(predicted);
  all_probs.push_back(probs);
  probs.clear();

  logic2on3(predictions, all_probs);

  imwrite("originale.png", image);
  imwrite("flippata_orizzontale.png", image_flip_o);
  imwrite("flippata_verticale.png", image_flip_v);*/


  delete caffe_test_net;


   return 0;
}
