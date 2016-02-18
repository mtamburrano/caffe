#include <mosquittopp.h>

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


class MqttCaffe : public mosqpp::mosquittopp {
 public:
  MqttCaffe(const char *id);
  ~MqttCaffe() {
    std::cout << "mqtt distrutto"<<std::endl;
  };

  void on_connect(int rc);
  void on_message(const struct mosquitto_message *message);
  void on_subscribe(int mid, int qos_count, const int *granted_qos);

  void on_disconnect(int rc);


  bool isShuffling() {
    return IS_SHUFFLING_;
  }

 private:
  bool IS_SHUFFLING_;
  std::string msg_from_generator_;

};
