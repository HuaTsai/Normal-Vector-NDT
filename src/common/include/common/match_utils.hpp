#pragma once

#include <bits/stdc++.h>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Dense>
#include <gsl/gsl>

namespace common {
class RandomTransformGenerator2D {
 public:
  explicit RandomTransformGenerator2D(bool fix_seed)
      : is_radius_set_(false),
        is_angle_set_(false),
        radius_min_(0),
        radius_max_(0),
        angle_rad_min_(0),
        angle_rad_max_(0),
        center_x_(0),
        center_y_(0),
        center_th_(0) {
    if (fix_seed) {
      dre_ = std::make_shared<std::default_random_engine>(1);
    } else {
      std::random_device rd;
      dre_ = std::make_shared<std::default_random_engine>(rd());
    }
  }

  void SetTranslationRadiusBound(double min, double max) {
    Expects(min >= 0 && max > min);
    radius_min_ = min;
    radius_max_ = max;
    is_radius_set_ = true;
  }

  std::vector<double> GetTranslationRadiusBound() const {
    return {radius_min_, radius_max_};
  }

  void SetRotationDegreeBound(double min, double max) {
    SetRotationRadianBound(min * M_PI / 180., max * M_PI / 180.);
  }

  void SetRotationRadianBound(double min, double max) {
    Expects(min >= 0 && max > min);
    angle_rad_min_ = min;
    angle_rad_max_ = max;
    is_angle_set_ = true;
  }

  std::vector<double> GetRotationDegreeBound() const {
    return {angle_rad_min_ * 180. / M_PI, angle_rad_max_ * 180. / M_PI};
  }

  void SetCenterXYDegree(double x, double y, double degree) {
    SetCenterXYRadian(x, y, degree * M_PI / 180.);
  }

  void SetCenterXYRadian(double x, double y, double radian) {
    center_x_ = x;
    center_y_ = y;
    center_th_ = radian;
  }

  void Clear() {
    is_radius_set_ = false;
    is_angle_set_ = false;
    radius_min_ = 0;
    radius_max_ = 0;
    angle_rad_min_ = 0;
    angle_rad_max_ = 0;
    center_x_ = 0;
    center_y_ = 0;
    center_th_ = 0;
  }

  std::vector<Eigen::Matrix3d> Generate(size_t sizes) {
    Expects(is_radius_set_ && is_angle_set_);
    std::vector<Eigen::Matrix3d> ret(sizes, Eigen::Matrix3d::Identity());
    std::uniform_real_distribution<> radius_urd(radius_min_ * radius_min_,
                                                radius_max_ * radius_max_);
    std::uniform_real_distribution<> theta_urd(0, 2 * M_PI);
    std::uniform_real_distribution<> angle_urd(angle_rad_min_, angle_rad_max_);
    for (size_t i = 0; i < sizes; ++i) {
      double radius = sqrt(radius_urd(*dre_));
      double theta = theta_urd(*dre_);
      double angle = angle_urd(*dre_);
      ret.at(i)(0, 2) = center_x_ + radius * cos(theta);
      ret.at(i)(1, 2) = center_y_ + radius * sin(theta);
      ret.at(i)(0, 0) = cos(center_th_ + angle);
      ret.at(i)(0, 1) = -sin(center_th_ + angle);
      ret.at(i)(1, 0) = sin(center_th_ + angle);
      ret.at(i)(1, 1) = cos(center_th_ + angle);
    }
    return ret;
  }

 private:
  std::shared_ptr<std::default_random_engine> dre_;
  bool is_radius_set_;
  bool is_angle_set_;
  double radius_min_;
  double radius_max_;
  double angle_rad_min_;
  double angle_rad_max_;
  double center_x_;
  double center_y_;
  double center_th_;
};

struct MatchPackage {
  MatchPackage()
      : guess(Eigen::Matrix3d::Identity()),
        result(Eigen::Matrix3d::Identity()),
        actual(Eigen::Matrix3d::Identity()),
        iters(0) {}
  Eigen::MatrixXd source;
  Eigen::MatrixXd target;
  Eigen::MatrixXd output;
  Eigen::Matrix3d guess;
  Eigen::Matrix3d result;
  Eigen::Matrix3d actual;
  int iters;
};

struct Correspondences {
  Correspondences() {}
  Eigen::MatrixXd mtx;
  void PushBack(const Eigen::Vector3d &from,
                const Eigen::Vector3d &to,
                double score) {
    mtx.conservativeResize(7, mtx.cols() + 1);
    mtx.block<3, 1>(0, mtx.cols() - 1) = from;
    mtx.block<3, 1>(3, mtx.cols() - 1) = to;
    mtx(6, mtx.cols() - 1) = score;
  }
};

class MatchInternal {
 public:
  MatchInternal() : has_data_(false), cell_size_(0) {}

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize_mtx(Archive &ar, std::string str, Eigen::MatrixXd &mtx) {
    int row, col;
    if (has_data_) {
      row = mtx.rows();
      col = mtx.cols();
    }
    ar & boost::serialization::make_nvp((str + "_r").c_str(), row);
    ar & boost::serialization::make_nvp((str + "_c").c_str(), col);
    if (!has_data_) {
      mtx.setZero(row, col);
    }
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        ar & boost::serialization::make_nvp(str.c_str(), mtx(i, j));
      }
    }
  }

  template <typename Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("comment", comment_);
    ar & boost::serialization::make_nvp("cell_size", cell_size_);
    serialize_mtx(ar, "source", source_);
    serialize_mtx(ar, "target", target_);
    int total_iters;
    if (has_data_) {
      total_iters = corrs_.size();
    }
    ar & BOOST_SERIALIZATION_NVP(total_iters);
    if (!has_data_) {
      corrs_.resize(total_iters);
      tfs_.resize(total_iters);
    }
    for (int i = 0; i < total_iters; ++i) {
      std::string corrsi("corrs_" + std::to_string(i));
      std::string tfsi("tfs_" + std::to_string(i));
      serialize_mtx(ar, corrsi, corrs_.at(i));
      ar & boost::serialization::make_nvp(tfsi.c_str(), tfs_.at(i)(0));
      ar & boost::serialization::make_nvp(tfsi.c_str(), tfs_.at(i)(1));
      ar & boost::serialization::make_nvp(tfsi.c_str(), tfs_.at(i)(2));
    }
    has_data_ = true;
  }

  void PushBack(const Correspondences &corr, const Eigen::Vector3d &tf) {
    corrs_.push_back(corr.mtx);
    tfs_.push_back(tf);
    has_data_ = true;
  }

  void PushBack(const Correspondences &corr, const std::vector<double> &tf) {
    Expects(tf.size() == 3);
    PushBack(corr, Eigen::Vector3d::Map(tf.data(), 3));
  }

  void ClearResults() {
    corrs_.clear();
    tfs_.clear();
    has_data_ = false;
  }

  bool has_data() const { return has_data_; }
  double cell_size() const { return cell_size_; }
  Eigen::MatrixXd source() const { return source_; }
  Eigen::MatrixXd target() const { return target_; }
  std::vector<Eigen::MatrixXd> corrs() const { return corrs_; }
  std::vector<Eigen::Vector3d> tfs() const { return tfs_; }
  std::string comment() const { return comment_; }

  void set_has_data(bool has_data) { has_data_ = has_data; }
  void set_cell_size(double cell_size) { cell_size_ = cell_size; }
  void set_source(const Eigen::MatrixXd &source) { source_ = source; }
  void set_target(const Eigen::MatrixXd &target) { target_ = target; }
  void set_corrs(const std::vector<Eigen::MatrixXd> &corrs) { corrs_ = corrs; }
  void set_tfs(const std::vector<Eigen::Vector3d> &tfs) { tfs_ = tfs; }
  void set_comment(const std::string &comment) { comment_ = comment; }

 private:
  bool has_data_;
  double cell_size_;
  Eigen::MatrixXd source_;
  Eigen::MatrixXd target_;
  std::vector<Eigen::MatrixXd> corrs_;
  std::vector<Eigen::Vector3d> tfs_;
  std::string comment_;
};

class MatchResults {
 public:
  MatchResults() : has_data_(false) {}

  // void SetSourceTarget(const Eigen::MatrixXd &source, const Eigen::MatrixXd &target) {
  //   source_ = source;
  //   target_ = target;
  //   has_data_ = true;
  // }

  // void SetActualTF(const std::vector<double> &actual) {
  //   actual_ = actual;
  // }

  void PushBack(const Eigen::Vector3d &guess,
                const Eigen::Vector3d &result,
                double score) {
    guesses_.conservativeResize(3, guesses_.cols() + 1);
    guesses_.col(guesses_.cols() - 1) = guess;
    results_.conservativeResize(3, results_.cols() + 1);
    results_.col(results_.cols() - 1) = result;
    scores_.push_back(score);
  }

  void PushBack(const std::vector<double> &guess,
                const std::vector<double> &result,
                double score) {
    Expects(guess.size() == 3);
    Expects(result.size() == 3);
    PushBack(Eigen::Vector3d::Map(guess.data(), 3),
             Eigen::Vector3d::Map(result.data(), 3), score);
  }

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize_mtx(Archive &ar, std::string str, Eigen::MatrixXd &mtx) {
    int row, col;
    if (has_data_) {
      row = mtx.rows();
      col = mtx.cols();
    }
    ar & boost::serialization::make_nvp((str + "_r").c_str(), row);
    ar & boost::serialization::make_nvp((str + "_c").c_str(), col);
    if (!has_data_) {
      mtx.setZero(row, col);
    }
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        ar & boost::serialization::make_nvp(str.c_str(), mtx(i, j));
      }
    }
  }

  template <typename Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("comment", comment_);
    serialize_mtx(ar, "source", source_);
    serialize_mtx(ar, "target", target_);
    serialize_mtx(ar, "guesses", guesses_);
    serialize_mtx(ar, "results", results_);
    ar & boost::serialization::make_nvp("actual", actual_);
    ar & boost::serialization::make_nvp("scores", scores_);
    has_data_ = true;
  }

  Eigen::MatrixXd source() const { return source_; }
  Eigen::MatrixXd target() const { return target_; }
  Eigen::MatrixXd guesses() const { return guesses_; }
  Eigen::MatrixXd results() const { return results_; }
  std::vector<double> actual() const { return actual_; }
  std::vector<double> scores() const { return scores_; }
  std::string comment() const { return comment_; }
  
  void set_has_data(bool has_data) { has_data_ = has_data; }
  void set_source(const Eigen::MatrixXd &source) { source_ = source; }
  void set_target(const Eigen::MatrixXd &target) { target_ = target; }
  void set_actual(const std::vector<double> &actual) { actual_ = actual; }
  void set_comment(const std::string &comment) { comment_ = comment; }

 private:
  bool has_data_;
  Eigen::MatrixXd source_;
  Eigen::MatrixXd target_;
  Eigen::MatrixXd results_;
  Eigen::MatrixXd guesses_;
  std::vector<double> actual_;
  std::vector<double> scores_;
  std::string comment_;
  // std::vector<int> states_;
};

template <typename T>
void SerializeOut(std::string file_name, const T &data) {
  std::ofstream ofs(file_name);
  boost::archive::xml_oarchive oa(ofs);
  oa << BOOST_SERIALIZATION_NVP(data);
}

template <typename T>
void SerializeIn(std::string file_name, T &data) {
  std::ifstream ifs(file_name);
  boost::archive::xml_iarchive ia(ifs);
  ia >> BOOST_SERIALIZATION_NVP(data);
}

std::vector<std::vector<double>> Combinations(std::vector<std::vector<double>> x) {
  /* input: {{0, 1}, {a, b}, {x, y}}
   * output: {{0, a, x}, {0, a, y}, ..., {1, b, y}}
   */
  std::vector<std::vector<double>> ret;
  std::vector<std::vector<double>::iterator> its;
  for (size_t i = 0; i < x.size(); ++i) {
    its.push_back(x.at(i).begin());
  }
  while (its.at(0) != x.at(0).end()) {
    std::vector<double> tmp;
    for (size_t i = 0; i < x.size(); ++i) {
      tmp.push_back(*its.at(i));
    }
    ret.push_back(tmp);
    ++its.back();
    for (size_t i = x.size() - 1; (i > 0) && (its.at(i) == x.at(i).end()); --i) {
      its.at(i) = x.at(i).begin();
      ++its.at(i - 1);
    }
  }
  return ret;
}

}  // namespace common
