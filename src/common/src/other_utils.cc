#include <common/other_utils.h>

#include <boost/filesystem.hpp>

std::pair<double, double> ComputeMeanAndStdev(const std::vector<double> &coll) {
  if (coll.size() <= 1) {
    std::cerr << __FUNCTION__ << ": invalid container size " << coll.size()
              << ", return with zero standard deviation\n";
    return {coll.size() ? coll[0] : 0., 0.};
  }
  double mean = std::accumulate(coll.begin(), coll.end(), 0.) / coll.size();
  double stdev = std::sqrt(accumulate(coll.begin(), coll.end(), 0.,
                                      [&mean](auto a, auto b) {
                                        return a + (b - mean) * (b - mean);
                                      }) /
                           (coll.size() - 1));
  return {mean, stdev};
}

std::string GetCurrentTimeAsString() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
  return oss.str();
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
  std::exit(1);
  return {};
}

// Boost solution: can be replaced by filesystem in C++17
std::vector<std::string> GetBagsPath(int lognum) {
  std::vector<std::string> ret;
  std::regex reg(".*log" + std::to_string(lognum) + "_.*\\.bag");
  boost::filesystem::directory_iterator it(BPATH);
  boost::filesystem::directory_iterator end;
  while (it != end) {
    if (is_regular_file(it->path()) &&
        std::regex_match(it->path().string(), reg))
      ret.push_back(it->path().string());
    ++it;
  }
  std::sort(ret.begin(), ret.end());
  return ret;
}

std::string GetScenePath(int scenenum) {
  std::regex reg(".*-0?" + std::to_string(scenenum) + "\\.bag");
  boost::filesystem::directory_iterator it(BPATH);
  boost::filesystem::directory_iterator end;
  while (it != end) {
    if (is_regular_file(it->path()) &&
        std::regex_match(it->path().string(), reg))
      return it->path().string();
    ++it;
  }
  return "";
}

std::vector<int> LargestNIndices(const std::vector<double> &data, int n) {
  std::multimap<double, int, std::greater<>> mp;
  for (size_t i = 0; i < data.size(); ++i) mp.insert({data[i], i});
  std::vector<int> ret;
  for (int i = 0; i < n; ++i) ret.push_back(std::next(mp.begin(), i)->second);
  return ret;
}
