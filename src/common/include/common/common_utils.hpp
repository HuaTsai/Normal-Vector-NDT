#pragma once

#include <bits/stdc++.h>
#include <ros/ros.h>

#define APATH "/home/ee904/Desktop/HuaTsai/NormalNDT/Analysis"

inline std::chrono::steady_clock::time_point GetTime() {
  return std::chrono::steady_clock::now();
}

inline int GetDiffTime(std::chrono::steady_clock::time_point t1,
                       std::chrono::steady_clock::time_point t2) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
}

std::string JoinPath() {
  return "";
}

template <typename... Args>
std::string JoinPath(std::string s1, Args... args) {
  std::string s2 = JoinPath(args...);
  if (s1.size() && s1.back() == '/') s1.pop_back();
  if (!s1.size()) return s2;
  if (s2.size() && s2[0] == '/') s2.erase(s2.begin());
  if (!s2.size()) return s1;
  return s1 + "/" + s2;
}

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

std::string GetDataPath(std::string data) {
  return JoinPath(APATH, "1Data");
}

std::vector<std::string> GetBagsPath(std::string data) {
  std::vector<std::string> ret;
  if (data == "log24") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729278446231_scene-0299.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729298446271_scene-0300.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729318549677_scene-0301.bag");
  } else if (data == "log62-1") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193241547892_scene-0997.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193261546825_scene-0998.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193281648047_scene-0999.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193301547950_scene-1000.bag");
  } else if (data == "log62-2") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193461547574_scene-1004.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193481898177_scene-1005.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193501549291_scene-1006.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193521798725_scene-1007.bag");
  } else {
    std::cerr << "No specified data " << data << std::endl;
    exit(-1);
  }
  return ret;
}
