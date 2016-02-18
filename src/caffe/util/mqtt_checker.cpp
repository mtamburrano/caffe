#include "caffe/util/mqtt_checker.hpp"
#include "caffe/common.hpp"


MqttCaffe::MqttCaffe(const char *id)
    : mosquittopp(id) {
  int keepalive = 60;
  int port = 1883;
  std::string host = "localhost";
  /* Connect immediately. This could also be done by calling
   * mqtt_tempconv->connect(). */
  IS_SHUFFLING_ = false;
  connect(host.c_str(), port, keepalive);
}

void MqttCaffe::on_connect(int rc) {
  LOG(INFO) << "MQQT Connected with code: " << rc;
  if (rc == 0) {
    /* Only attempt to subscribe on a successful connect. */
    subscribe(NULL, "shuffle_news");
  }
}
void MqttCaffe::on_message(const struct mosquitto_message *message) {
  int size_msg = message->payloadlen + 1;  //+1 perchè la stringa terminerà con 0
  char* buf = (char*) malloc(size_msg);

  memset(buf, 0, size_msg * sizeof(char));
  memcpy(buf, message->payload, (size_msg - 1) * sizeof(char));  //-1 così la stringa terminerà on 0 (già messo dalla memset)

  if (!strcmp(message->topic, "shuffle_news")) {
    msg_from_generator_ = std::string(buf);
  }

  if(msg_from_generator_ == "NEED SHUFFLE")
  {
    IS_SHUFFLING_ = true;
    LOG(INFO) <<  "Generator wants to shuffle, stop to train NOW!"<<std::endl;
  }
  if(msg_from_generator_ == "SHUFFLE DONE")
  {
    IS_SHUFFLING_ = false;
  }

  free(buf);
}
void MqttCaffe::on_subscribe(int mid, int qos_count, const int *granted_qos) {
  LOG(INFO) <<  "Subscription succeeded";
}

void MqttCaffe::on_disconnect(int rc) {
  LOG(INFO) <<  std::endl << "MQQT DISCONNECTED";
}
