// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <signal.h>
#include <atomic>

#include "boost/scoped_ptr.hpp"
#include "boost/filesystem.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "lmdb.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db_lmdb.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_reader.hpp"

/////////AGGIUNTI PER IL GENERATOR/////////////
#include <memory>
#include "caffe/data_layers.hpp"
#include "caffe/net.hpp"
#include <boost/function.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <datasetgenerator/DataGenerator.hpp>
#include <datasetgenerator/AutoveloxDataGenerator.hpp>
//////////////////////////////////////////////

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, true,
    "When this option is on, treat images as grayscale ones");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");


volatile sig_atomic_t quit = 0;

void got_signal(int s)
{
  if(s == SIGINT)
  {
    std::cout << "segnale di stop ricevuto, commito gli ultimi update prima di chiudere" << std::endl;
    quit = s;
  }
}

int s2i(std::string s)
{
  std::stringstream ss;
  ss << s;
  int i;
  ss >> i;
  return i;
}

std::string vectorToString(std::vector<std::string> vector_string) {

  std::string ret = "";
  for(int i = 0; i<vector_string.size(); ++i)
    ret = ret + vector_string[i];
  return ret;
}

void vectorIntToDatum(std::vector<int> vector_int, Datum* datum) {

  datum->set_channels(vector_int.size());
  datum->set_height(1);
  datum->set_width(1);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  for(int i = 0; i< vector_int.size(); ++i)
  {
    //uso float_data perchè il campo data di Datum è uint8, così potrebbe contenere solo i
    //valori 0...255
    datum->add_float_data(static_cast< float >( vector_int[i]));
  }
}


void datumToCVMat(const Datum* datum, cv::Mat& cv_img) {
  if(datum->channels() == 1)
    cv_img.create(datum->height(), datum->width(), CV_8UC1);
  else
    cv_img.create(datum->height(), datum->width(), CV_8UC3);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();

  int datum_index = 0;
  for (int h = 0; h < datum_height; ++h) {
    uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        ptr[img_index] = static_cast<unsigned char>(datum->data()[datum_index]);
        img_index++;
        datum_index++;
      }
    }
  }
}


std::string readDatumLabelValue(const Datum* datum_label, std::unique_ptr<caffe::AutoveloxDataGenerator>* datagenerator)
{
  string label_value = "";
  for(int d = 0; d < datum_label->float_data_size(); ++d)
  {
    int fromClass = static_cast<int>(datum_label->float_data(d));
    label_value += (*datagenerator)->getLabelFromClass(fromClass);
  }
  return label_value;
}



////////////////////////////////////

        ///READ

      //vengono lette num_read_images. Se num_read_images > delle immagini del database, ricomincia dall'inizio

///////////////////////////////////


void read(string db_data_name, string db_label_name, int num_read_images)
{
  //controllo se il database già esiste
  boost::filesystem::path path_db(db_data_name);
  if(!boost::filesystem::exists(path_db))
  {
    std::cout<<"il database \""<<db_data_name<<"\" non esiste, che leggo?"<<std::endl;
    return;
  }

  std::unique_ptr<caffe::AutoveloxDataGenerator> datagenerator;
  datagenerator = std::unique_ptr<caffe::AutoveloxDataGenerator>(new caffe::AutoveloxDataGenerator());
  datagenerator->init();

  // Leggo dai DB dati e label
  scoped_ptr<db::DB> db_data(db::GetDB(FLAGS_backend));
  scoped_ptr<db::DB> db_label(db::GetDB(FLAGS_backend));

  db_data->Open(db_data_name.c_str(), db::READ);
  db_label->Open(db_label_name.c_str(), db::READ);

  int db_size = db_data->GetSize();
  std::cout << "DB SIZE: "<<db_size<<std::endl;

  shared_ptr<db::Cursor> cursor_data(db_data->NewCursor());
  shared_ptr<db::Cursor> cursor_label(db_label->NewCursor());

  // Reading from to db
  cv::Mat img;
  Datum* datum_data = new Datum();
  Datum* datum_label = new Datum();
  int index_db = 0;
  for(int ni = 0; ni < num_read_images; ++ni)
  {

    datum_data->ParseFromString(cursor_data->value());
    datum_label->ParseFromString(cursor_label->value());

    //mostro l'immagine e la label
    datumToCVMat(datum_data, img);
    std::string label_value = readDatumLabelValue(datum_label, &datagenerator);
    std::cout << index_db << ") Label: " << label_value << std::endl;
    cv::imshow("img", img);
    cv::waitKey();

    // go to the next iter

    cursor_data->Renew();
    cursor_label->Renew();
    cursor_data->Next();
    cursor_label->Next();
    index_db++;

    if (!cursor_data->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_data->SeekToFirst();
      cursor_label->SeekToFirst();
      index_db = 0;
    }
  }
  std::cout << "DB SIZE: "<<db_size<<std::endl;

  delete datum_data;
  delete datum_label;

}

////////////////////////////////////


////////////////////////////////////

        ///CREATE

    //vengono create num_generated_images immagini

///////////////////////////////////


void create(string db_data_name, string db_label_name, int num_generated_images)
{
  //controllo se il database già esiste
  boost::filesystem::path path_db(db_data_name);
  if(boost::filesystem::exists(path_db))
  {
    std::string input;
    std::cout<<"il database \""<<db_data_name<<"\" già esiste, vuoi sovrascriverlo? (y/n)"<<std::endl;
    if (!std::getline(std::cin, input)) { /* error, abort! */ }
    if(input == "y")
    {
      boost::filesystem::remove_all(db_data_name);
      boost::filesystem::remove_all(db_label_name);
    }
    else
    {
      std::cout<<"creazione annullata"<<std::endl;
      return;
    }
  }else
  {
    std::cout << "Verranno creati i db \""<<db_data_name<<"\" e \""<<db_label_name<<"\""<<std::endl;
  }

  std::unique_ptr<caffe::AutoveloxDataGenerator> datagenerator;
  datagenerator = std::unique_ptr<caffe::AutoveloxDataGenerator>(new caffe::AutoveloxDataGenerator());
  datagenerator->init();

  //vettore che contiene le coppie mat,label, dove la label è un vettore di stringhe
  // (ogni stringa rappresenta però un intero, ad esempio "3")
  std::vector<std::pair<cv::Mat, std::vector<std::string> > > lines;

  //creo immagini e label
  std::vector<cv::Mat> renderMats;
  std::vector<std::string> label_string;
  for(int ngi = 0; ngi < num_generated_images; ++ngi)
  {
    if(renderMats.size() > 0) {
      renderMats.clear();
      label_string.clear();
    }
    datagenerator->render(renderMats);
    datagenerator->getLabel(label_string);

    lines.push_back(std::make_pair(renderMats[0], label_string));

    if( ngi % 1000 == 0) {
      LOG(INFO) << "Generated " << ngi << " images.";
    }
  }


  LOG(INFO) << "A total of " << lines.size() << " images.";

  // Creo un DB per i dati ed uno per le label
  scoped_ptr<db::DB> db_data(db::GetDB(FLAGS_backend));
  scoped_ptr<db::DB> db_label(db::GetDB(FLAGS_backend));

  db_data->Open(db_data_name.c_str(), db::NEW);
  db_label->Open(db_label_name.c_str(), db::NEW);
  scoped_ptr<db::Transaction> txn_data(db_data->NewTransaction());
  scoped_ptr<db::Transaction> txn_label(db_label->NewTransaction());

  // Storing to db
  Datum datum_data;
  Datum datum_label;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;

    //creo il Datum per l'immagine
    CVMatToDatum(lines[line_id].first, &datum_data);
    //inserisco una fake label perchè non mi interessa, dovrò creare un database solo per le label dopo
    datum_data.set_label(-17);

    //creo il Datum per la label
    std::vector<int> label_int;
    std::vector<std::string> label_string = lines[line_id].second;

    for(int nc = 0; nc < datagenerator->getNumberOfChars(); ++nc) {
      label_int.push_back(datagenerator->getClass(label_string[nc]));
    }
    vectorIntToDatum(label_int, &datum_label);
    //inserisco una fake label perchè non mi interessa la label della label °O°
    datum_label.set_label(-18);


    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%09d_%s", line_id,
            "key");

    // Put in db
    string out_data;
    string out_label;
    CHECK(datum_data.SerializeToString(&out_data));
    CHECK(datum_label.SerializeToString(&out_label));
    std::cout << "key_cstr: "<<key_cstr<<" - value: "<<vectorToString(label_string)<<std::endl;
    //cv::imshow("prima", lines[line_id].first);
    //cv::waitKey();
    txn_data->Put(string(key_cstr, length), out_data);
    txn_label->Put(string(key_cstr, length), out_label);

    if (++count % 1000 == 0) {
      // Commit db
      txn_data->Commit();
      txn_data.reset(db_data->NewTransaction());
      txn_label->Commit();
      txn_label.reset(db_label->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn_data->Commit();
    txn_label->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  return;
}

////////////////////////////////////

        ///UPDATE TOT ALLA VOLTA THREAD SAFE (FORSE)

        //vengono modificate num_images_replaced_at_once alla volta

///////////////////////////////////

void update(string db_data_name, string db_label_name, int num_images_replaced_at_once)
{
  //controllo se il database già esiste
  boost::filesystem::path path_db(db_data_name);
  if(!boost::filesystem::exists(path_db))
  {
    std::cout<<"il database \""<<db_data_name<<"\" non esiste, che aggiorno?"<<std::endl;
    return;
  }

  std::unique_ptr<caffe::AutoveloxDataGenerator> datagenerator;
  datagenerator = std::unique_ptr<caffe::AutoveloxDataGenerator>(new caffe::AutoveloxDataGenerator());
  datagenerator->init();

  //intercetto Ctrl+c per completare l'ultimo update prima di uscire
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = got_signal;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  // Apro un DB per i dati ed uno per le label
  scoped_ptr<db::DB> db_data(db::GetDB(FLAGS_backend));
  scoped_ptr<db::DB> db_label(db::GetDB(FLAGS_backend));

  db_data->Open(db_data_name.c_str(), db::WRITE);
  db_label->Open(db_label_name.c_str(), db::WRITE);

  // Storing to db
  Datum datum_data;
  Datum datum_label;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;

  //shared_ptr<db::Cursor> cursor_data(db_data->NewCursor());
  //shared_ptr<db::Cursor> cursor_label(db_label->NewCursor());
  int index_db = 0;
  bool need_reset = false;

  int db_size = db_data->GetSize();

  scoped_ptr<db::Transaction> txn_data(db_data->NewTransaction());
  scoped_ptr<db::Transaction> txn_label(db_label->NewTransaction());

  int cycles_done = 0;

  while(true)
  {
    //vettore che contiene le coppie mat,label, dove la label è un vettore di stringhe
    // (ogni stringa rappresenta però un intero, ad esempio "3")
    std::vector<std::pair<cv::Mat, std::vector<std::string> > > lines;

    //creo immagini e label
    std::vector<cv::Mat> renderMats;
    std::vector<std::string> label_string;

    for(int ngi = 0; ngi < num_images_replaced_at_once; ++ngi)
    {
      if(renderMats.size() > 0) {
        renderMats.clear();
        label_string.clear();
      }
      datagenerator->render(renderMats);
      datagenerator->getLabel(label_string);

      //creo il Datum per l'immagine
      CVMatToDatum(renderMats[0], &datum_data);
      //inserisco una fake label perchè non mi interessa, dovrò creare un database solo per le label dopo
      datum_data.set_label(-17);

      //creo il Datum per la label
      std::vector<int> label_int;

      for(int nc = 0; nc < datagenerator->getNumberOfChars(); ++nc) {
        label_int.push_back(datagenerator->getClass(label_string[nc]));
      }
      vectorIntToDatum(label_int, &datum_label);
      //inserisco una fake label perchè non mi interessa la label della label °O°
      datum_label.set_label(-18);


      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%09d_%s", index_db,
                  "key");

      // Put in db
      string out_data;
      string out_label;
      CHECK(datum_data.SerializeToString(&out_data));
      CHECK(datum_label.SerializeToString(&out_label));

      txn_data->Put(string(key_cstr, length), out_data);
      txn_label->Put(string(key_cstr, length), out_label);

      //cv::imshow("rimpiazzo", renderMats[0]);
      //cv::waitKey();

      index_db++;
      if (index_db >= db_size) {
        index_db = 0;
        cycles_done++;
        std::cout << "CICLO COMPLETATO" << std::endl;
      }

    }

    txn_data->Commit();
    txn_data.reset(db_data->NewTransaction());
    txn_label->Commit();
    txn_label.reset(db_label->NewTransaction());
    std::cout <<std::endl<< "COMMITTED "<<(cycles_done*db_size)+index_db<<std::endl;
    //sleep(1);

    if(quit == SIGINT)
    {
      break;
    }

  }
  return;

}



int main(int argc, char** argv) {

  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Create, read or update a leveldb/lmdb database\n"
        "Usage:\n"
        "    lmbdGenerator [FLAGS] ACTION{create, update, read} DB_NAME NUM_IMAGES\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/lmdbGenerator");
    return 1;
  }
  std::string action = argv[1];
  if (action != "create" && action != "update" && action != "read")
  {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/lmdbGenerator");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  string num_images = argv[3];
  const int num_images_int = s2i(num_images);


  string db_data_name(argv[2]);
  db_data_name += "_data";
  string db_label_name(argv[2]);
  db_label_name += "_label";

  if(action == "create")
  {
    //vengono create num_images_int immagini
    create(db_data_name, db_label_name, num_images_int);
  }
  else if(action == "update")
  {
    //vengono modificate num_images_int immagini alla volta
    update(db_data_name, db_label_name, num_images_int);
  }
  else if(action == "read")
  {
    //vengono lette num_images_int. Se num_images_int > delle immagini del database, ricomincia dall'inizio
    read(db_data_name, db_label_name, num_images_int);
  }
  else
  {
    std::cout << "Azione non prevista. Azioni supportate: create, read, update" << std::endl;
  }

}
