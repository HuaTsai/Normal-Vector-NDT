#pragma once

#include <bits/stdc++.h>
#include <ros/ros.h>

#define APATH(name) "/home/ee904/Desktop/HuaTsai/NormalNDT/Analysis/"#name

void dprintf(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  size_t required = 1 + vsnprintf(NULL, 0, format, ap);
  va_end(ap);
  char *buf = new char[required];
  va_start(ap, format);
  vsnprintf(buf, required, format, ap);
  va_end(ap);
  fprintf(stderr, "%s", buf);
  fflush(stderr);
  delete[] buf;
}

namespace common {
template <typename T>
void SerializationInput(const std::string &filepath, T &msg) {
  std::ifstream ifs(filepath, std::ios::in | std::ios::binary);
  ifs.seekg(0, std::ios::end);
  std::streampos end = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::streampos begin = ifs.tellg();
  uint32_t size = end - begin;
  boost::shared_array<uint8_t> buffer(new uint8_t[size]);
  ifs.read((char *)buffer.get(), size);
  ros::serialization::IStream stream(buffer.get(), size);
  ros::serialization::deserialize(stream, msg);
  ifs.close();
}

template <typename T>
void SerializationOutput(const std::string &filepath, const T &msg) {
  uint32_t size = ros::serialization::serializationLength(msg);
  boost::shared_array<uint8_t> buffer(new uint8_t[size]);
  ros::serialization::OStream stream(buffer.get(), size);
  ros::serialization::serialize(stream, msg);
  std::ofstream ofs(filepath, std::ios::out | std::ios::binary);
  ofs.write((char *)buffer.get(), size);
  ofs.close();
}

std::vector<std::string> GetDataPath(std::string data) {
  std::vector<std::string> ret;
  if (data == "log24") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729278446231_scene-0299.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729298446271_scene-0300.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729318549677_scene-0301.bag");
  }
  return ret;
}

// void WriteToFile(std::string filepath, const std::vector<double> &data) {
//   std::ofstream ofs(filepath, std::ios::out);
//   for (const auto &elem : data) {
//     ofs << elem << " ";
//   }
//   ofs << std::endl;
//   ofs.close();
// }

// std::vector<double> ReadFromFile(std::string filepath) {
//   std::ifstream ifs(filepath, std::ios::in);
//   std::vector<double> ret;
//   double num;
//   while (ifs >> num) {
//     ret.push_back(num);
//   }
//   ifs.close();
//   return ret;
// }
}  // namespace common
