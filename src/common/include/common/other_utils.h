#pragma once
#include <bits/stdc++.h>
#include <ros/serialization.h>

// HACK: Change paths
#define APATH "/home/ee904/Desktop/HuaTsai/NormalNDT/Analysis"
#define BPATH "/home/ee904/Desktop/Dataset/nuScenes"
#define WSPATH "/home/ee904/Desktop/HuaTsai/NormalNDT/Research"
#define PYTHONPATH "/home/ee904/venv/com/bin/python"

std::pair<double, double> ComputeMeanAndStdev(const std::vector<double> &coll);

std::string GetCurrentTimeAsString();

std::string GetDataPath(std::string data);

std::vector<std::string> GetBagsPath(std::string data);

std::vector<std::string> GetBagsPath(int lognum);

std::string GetScenePath(int scenenum);

std::vector<int> LargestNIndices(const std::vector<double> &data, int n);

/********** Inline Functions **********/

inline std::chrono::steady_clock::time_point GetTime() {
  return std::chrono::steady_clock::now();
}

/** Get time difference in microseconds */
inline int GetDiffTime(std::chrono::steady_clock::time_point t1,
                       std::chrono::steady_clock::time_point t2) {
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

/********** Template functions **********/
template <typename T>
void Print(const std::vector<T> &data, std::string str = "") {
  if (str.size()) std::cout << str << ": ";
  for (size_t i = 0; i < data.size(); ++i)
    std::cout << data[i] << (i + 1 == data.size() ? "" : ", ");
  std::cout << std::endl;
}

inline std::string JoinPath() { return ""; }

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
