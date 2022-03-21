#pragma once
#include <ceres/autodiff_first_order_function.h>
#include <ceres/ceres.h>

#include <Eigen/Dense>

class NDTCost {
 public:
  NDTCost(const std::vector<Eigen::Vector3d> &ups,
          const std::vector<Eigen::Matrix3d> &cps,
          const std::vector<Eigen::Vector3d> &uqs,
          const std::vector<Eigen::Matrix3d> &cqs,
          double d2)
      : ups_(ups), cps_(cps), uqs_(uqs), cqs_(cqs), d2_(d2) {}
  template <typename T>
  bool operator()(const T *const tlrot, T *e) const {
    e[0] = T(0);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tlrot);
    Eigen::Map<const Eigen::Quaternion<T>> q(tlrot + 3);
    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<T, 3, 1> up = q * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 3, 3> cp = q * cps_[i].cast<T>() * q.conjugate();
      Eigen::Matrix<T, 3, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 3, 3> cq = cqs_[i].cast<T>();
      Eigen::Matrix<T, 3, 1> m = up - uq;
      Eigen::Matrix<T, 3, 3> c = cp + cq;
      e[0] += -ceres::exp(-d2_ * m.dot(c.inverse() * m));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector3d> &ups,
      const std::vector<Eigen::Matrix3d> &cps,
      const std::vector<Eigen::Vector3d> &uqs,
      const std::vector<Eigen::Matrix3d> &cqs,
      double d2) {
    return new ceres::AutoDiffFirstOrderFunction<NDTCost, 7>(
        new NDTCost(ups, cps, uqs, cqs, d2));
  }

 private:
  const std::vector<Eigen::Vector3d> ups_;
  const std::vector<Eigen::Matrix3d> cps_;
  const std::vector<Eigen::Vector3d> uqs_;
  const std::vector<Eigen::Matrix3d> cqs_;
  const double d2_;
};

class NNDTCost {
 public:
  NNDTCost(const std::vector<Eigen::Vector3d> &ups,
          const std::vector<Eigen::Matrix3d> &cps,
          const std::vector<Eigen::Vector3d> &nps,
          const std::vector<Eigen::Vector3d> &uqs,
          const std::vector<Eigen::Matrix3d> &cqs,
          const std::vector<Eigen::Vector3d> &nqs,
          double d2)
      : ups_(ups), cps_(cps), nps_(nps), uqs_(uqs), cqs_(cqs), nqs_(nqs), d2_(d2) {}
  template <typename T>
  bool operator()(const T *const tlrot, T *e) const {
    e[0] = T(0);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tlrot);
    Eigen::Map<const Eigen::Quaternion<T>> q(tlrot + 3);
    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<T, 3, 1> up = q * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 3, 3> cp = q * cps_[i].cast<T>() * q.conjugate();
      Eigen::Matrix<T, 3, 1> np = q * nps_[i].cast<T>();
      Eigen::Matrix<T, 3, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 3, 3> cq = cqs_[i].cast<T>();
      Eigen::Matrix<T, 3, 1> nq = nqs_[i].cast<T>();
      Eigen::Matrix<T, 3, 1> m1 = up - uq;
      Eigen::Matrix<T, 3, 1> m2 = (np + nq).normalized();
      Eigen::Matrix<T, 3, 3> c = cp + cq;
      e[0] += -ceres::exp(-d2_ * m1.dot(m2) * m1.dot(m2) / m2.dot(c * m2));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector3d> &ups,
      const std::vector<Eigen::Matrix3d> &cps,
      const std::vector<Eigen::Vector3d> &nps,
      const std::vector<Eigen::Vector3d> &uqs,
      const std::vector<Eigen::Matrix3d> &cqs,
      const std::vector<Eigen::Vector3d> &nqs,
      double d2) {
    return new ceres::AutoDiffFirstOrderFunction<NNDTCost, 7>(
        new NNDTCost(ups, cps, nps,  uqs, cqs, nqs, d2));
  }

 private:
  const std::vector<Eigen::Vector3d> ups_;
  const std::vector<Eigen::Matrix3d> cps_;
  const std::vector<Eigen::Vector3d> nps_;
  const std::vector<Eigen::Vector3d> uqs_;
  const std::vector<Eigen::Matrix3d> cqs_;
  const std::vector<Eigen::Vector3d> nqs_;
  const double d2_;
};