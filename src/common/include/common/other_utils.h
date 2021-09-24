#pragma once
#include <bits/stdc++.h>
#include <ros/ros.h>

// HACK: Change paths
#define APATH "/home/ee904/Desktop/HuaTsai/NormalNDT/Analysis"
#define BPATH "/home/ee904/Desktop/Dataset/nuScenes"
#define WSPATH "/home/ee904/Desktop/HuaTsai/NormalNDT/Research"
#define PYTHONPATH "/home/ee904/venv/com/bin/python"

inline std::chrono::steady_clock::time_point GetTime() {
  return std::chrono::steady_clock::now();
}

inline int GetDiffTime(std::chrono::steady_clock::time_point t1,
                       std::chrono::steady_clock::time_point t2) {
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
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
  return JoinPath(APATH, "1Data", data);
}

std::vector<std::string> GetBagsPath(std::string data) {
  if (data == "log24") {
    return {JoinPath(BPATH, "log24_1535729278446231_scene-0299.bag"),
            JoinPath(BPATH, "log24_1535729298446271_scene-0300.bag"),
            JoinPath(BPATH, "log24_1535729318549677_scene-0301.bag")};
  } else if (data == "log35-1") {
    return {JoinPath(BPATH, "log35_1538985150297491_scene-0420.bag"),
            JoinPath(BPATH, "log35_1538985170297464_scene-0421.bag"),
            JoinPath(BPATH, "log35_1538985190298034_scene-0422.bag"),
            JoinPath(BPATH, "log35_1538985210297489_scene-0423.bag"),
            JoinPath(BPATH, "log35_1538985230298065_scene-0424.bag"),
            JoinPath(BPATH, "log35_1538985250297526_scene-0425.bag"),
            JoinPath(BPATH, "log35_1538985270298113_scene-0426.bag"),
            JoinPath(BPATH, "log35_1538985290297603_scene-0427.bag")};
  } else if (data == "log35-2") {
    return {JoinPath(BPATH, "log35_1538985420297980_scene-0428.bag"),
            JoinPath(BPATH, "log35_1538985440646955_scene-0429.bag"),
            JoinPath(BPATH, "log35_1538985460298586_scene-0430.bag"),
            JoinPath(BPATH, "log35_1538985480297488_scene-0431.bag"),
            JoinPath(BPATH, "log35_1538985500297521_scene-0432.bag"),
            JoinPath(BPATH, "log35_1538985520296978_scene-0433.bag"),
            JoinPath(BPATH, "log35_1538985540297552_scene-0434.bag"),
            JoinPath(BPATH, "log35_1538985560396558_scene-0435.bag"),
            JoinPath(BPATH, "log35_1538985589297594_scene-0436.bag")};
  } else if (data == "log35-3") {
    return {JoinPath(BPATH, "log35_1538985679448351_scene-0437.bag"),
            JoinPath(BPATH, "log35_1538985699297420_scene-0438.bag"),
            JoinPath(BPATH, "log35_1538985719298546_scene-0439.bag")};
  } else if (data == "log62-1") {
    return {JoinPath(BPATH, "log62_1542193241547892_scene-0997.bag"),
            JoinPath(BPATH, "log62_1542193261546825_scene-0998.bag"),
            JoinPath(BPATH, "log62_1542193281648047_scene-0999.bag"),
            JoinPath(BPATH, "log62_1542193301547950_scene-1000.bag")};
  } else if (data == "log62-2") {
    return {JoinPath(BPATH, "log62_1542193461547574_scene-1004.bag"),
            JoinPath(BPATH, "log62_1542193481898177_scene-1005.bag"),
            JoinPath(BPATH, "log62_1542193501549291_scene-1006.bag"),
            JoinPath(BPATH, "log62_1542193521798725_scene-1007.bag")};
  }
  std::cerr << "No specified data " << data << std::endl;
  std::exit(-1);
  return {};
}

std::string GetCurrentTimeAsString() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
  return oss.str();
}
