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
      : ups_(ups),
        cps_(cps),
        nps_(nps),
        uqs_(uqs),
        cqs_(cqs),
        nqs_(nqs),
        d2_(d2) {}
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
      // XXX: Which one?
      Eigen::Matrix<T, 3, 1> m2 = (np + nq).normalized();
      // Eigen::Matrix<T, 3, 1> m2 = np + nq;
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
        new NNDTCost(ups, cps, nps, uqs, cqs, nqs, d2));
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

class NDTCostTR {
 public:
  NDTCostTR(const Eigen::Vector3d &up,
            const Eigen::Matrix3d &cp,
            const Eigen::Vector3d &uq,
            const Eigen::Matrix3d &cq,
            double d2)
      : up_(up), cp_(cp), uq_(uq), cq_(cq), d2_(d2) {}
  template <typename T>
  bool operator()(const T *const tlrot, T *e) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tlrot);
    Eigen::Map<const Eigen::Quaternion<T>> q(tlrot + 3);
    Eigen::Matrix<T, 3, 1> up = q * up_.cast<T>() + t;
    Eigen::Matrix<T, 3, 3> cp = q * cp_.cast<T>() * q.conjugate();
    Eigen::Matrix<T, 3, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 3, 3> cq = cq_.cast<T>();
    Eigen::Matrix<T, 3, 1> m = up - uq;
    Eigen::Matrix<T, 3, 3> c = cp + cq;
    e[0] = T(1) - ceres::exp(-d2_ * m.dot(c.inverse() * m));
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &up,
                                     const Eigen::Matrix3d &cp,
                                     const Eigen::Vector3d &uq,
                                     const Eigen::Matrix3d &cq,
                                     double d2) {
    return new ceres::AutoDiffCostFunction<NDTCostTR, 1, 7>(
        new NDTCostTR(up, cp, uq, cq, d2));
  }

 private:
  const Eigen::Vector3d up_;
  const Eigen::Matrix3d cp_;
  const Eigen::Vector3d uq_;
  const Eigen::Matrix3d cq_;
  const double d2_;
};

class NNDTCostTR {
 public:
  NNDTCostTR(const Eigen::Vector3d &up,
             const Eigen::Matrix3d &cp,
             const Eigen::Vector3d &np,
             const Eigen::Vector3d &uq,
             const Eigen::Matrix3d &cq,
             const Eigen::Vector3d &nq,
             double d2)
      : up_(up), cp_(cp), np_(np), uq_(uq), cq_(cq), nq_(nq), d2_(d2) {}

  template <typename T>
  bool operator()(const T *const tlrot, T *e) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tlrot);
    Eigen::Map<const Eigen::Quaternion<T>> q(tlrot + 3);
    Eigen::Matrix<T, 3, 1> up = q * up_.cast<T>() + t;
    Eigen::Matrix<T, 3, 1> unp = q * np_.cast<T>();
    Eigen::Matrix<T, 3, 3> cp = q * cp_.cast<T>() * q.conjugate();
    Eigen::Matrix<T, 3, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 3, 1> unq = nq_.cast<T>();
    Eigen::Matrix<T, 3, 3> cq = cq_.cast<T>();
    Eigen::Matrix<T, 3, 1> m1 = up - uq;
    Eigen::Matrix<T, 3, 1> m2 = (unp + unq).normalized();
    Eigen::Matrix<T, 3, 3> c1 = cp + cq;
    e[0] = T(1) - ceres::exp(-d2_ * m1.dot(m2) * m1.dot(m2) / m2.dot(c1 * m2));
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &up,
                                     const Eigen::Matrix3d &cp,
                                     const Eigen::Vector3d &np,
                                     const Eigen::Vector3d &uq,
                                     const Eigen::Matrix3d &cq,
                                     const Eigen::Vector3d &nq,
                                     double d2) {
    return new ceres::AutoDiffCostFunction<NNDTCostTR, 1, 7>(
        new NNDTCostTR(up, cp, np, uq, cq, nq, d2));
  }

 private:
  const Eigen::Vector3d up_;
  const Eigen::Matrix3d cp_;
  const Eigen::Vector3d np_;
  const Eigen::Vector3d uq_;
  const Eigen::Matrix3d cq_;
  const Eigen::Vector3d nq_;

  const double d2_;
};