#include <ceres/autodiff_first_order_function.h>
#include <ceres/ceres.h>
#include <sndt/ndt_cell.h>
#include <sndt/sndt_cell.h>

#include <Eigen/Dense>

template <typename T>
Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw) {
  T cos_yaw = ceres::cos(yaw);
  T sin_yaw = ceres::sin(yaw);
  Eigen::Matrix<T, 2, 2> ret;
  ret << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return ret;
}

class ICPCostFunctor {
 public:
  ICPCostFunctor(const Eigen::Vector2d &p, const Eigen::Vector2d &q)
      : p_(p), q_(q) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> q = q_.cast<T>();

    // Note: (p - q).norm() works, but with warning message (nan Jacobian)
    e[0] = (p - q)[0];
    e[1] = (p - q)[1];
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &q) {
    return new ceres::AutoDiffCostFunction<ICPCostFunctor, 2, 3>(
        new ICPCostFunctor(p, q));
  }

 private:
  const Eigen::Vector2d p_, q_;
};

class SICPCostFunctor {
 public:
  SICPCostFunctor(const Eigen::Vector2d &p,
                  const Eigen::Vector2d &np,
                  const Eigen::Vector2d &q,
                  const Eigen::Vector2d &nq)
      : p_(p), np_(np), q_(q), nq_(nq) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> np = R * np_.cast<T>();
    Eigen::Matrix<T, 2, 1> q = q_.cast<T>();
    Eigen::Matrix<T, 2, 1> nq = nq_.cast<T>();
    Eigen::Matrix<T, 2, 1> npq = (np + nq).normalized();

    *e = (p - q).dot(npq);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &np,
                                     const Eigen::Vector2d &q,
                                     const Eigen::Vector2d &nq) {
    return new ceres::AutoDiffCostFunction<SICPCostFunctor, 1, 3>(
        new SICPCostFunctor(p, np, q, nq));
  }

 private:
  const Eigen::Vector2d p_, np_, q_, nq_;
};

// BUG: Generalized Optimizer is needed here.
class SICPCostFunctor2 {
 public:
  SICPCostFunctor2(const std::vector<Eigen::Vector2d> &ps,
                   const std::vector<Eigen::Vector2d> &nps,
                   const std::vector<Eigen::Vector2d> &qs,
                   const std::vector<Eigen::Vector2d> &nqs)
      : ps_(ps), nps_(nps), qs_(qs), nqs_(nqs) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    for (size_t i = 0; i < ps_.size(); ++i) {
      Eigen::Matrix<T, 2, 1> p = R * ps_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 1> np = R * nps_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> q = qs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> nq = nqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> npq = (np + nq).normalized();

      e[0] += (p - q).dot(npq);
    }
    return true;
  }

  static ceres::CostFunction *Create(const std::vector<Eigen::Vector2d> &ps,
                                     const std::vector<Eigen::Vector2d> &nps,
                                     const std::vector<Eigen::Vector2d> &qs,
                                     const std::vector<Eigen::Vector2d> &nqs) {
    return new ceres::AutoDiffCostFunction<SICPCostFunctor2, 1, 3>(
        new SICPCostFunctor2(ps, nps, qs, nqs));
  }

 private:
  const std::vector<Eigen::Vector2d> ps_, nps_, qs_, nqs_;
};

class D2DNDTCostFunctor {
 public:
  D2DNDTCostFunctor(const std::vector<Eigen::Vector2d> &ups,
                    const std::vector<Eigen::Matrix2d> &cps,
                    const std::vector<Eigen::Vector2d> &uqs,
                    const std::vector<Eigen::Matrix2d> &cqs,
                    double d2)
      : ups_(ups), cps_(cps), uqs_(uqs), cqs_(cqs), d2_(d2) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<T, 2, 1> up = R * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 2> cp = R * cps_[i].cast<T>() * R.transpose();
      Eigen::Matrix<T, 2, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> m = up - uq;
      Eigen::Matrix<T, 2, 2> c = cp + cq;
      e[0] += -ceres::exp(-d2_ * m.dot(c.inverse() * m));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs,
      double d2) {
    return new ceres::AutoDiffFirstOrderFunction<D2DNDTCostFunctor, 3>(
        new D2DNDTCostFunctor(ups, cps, uqs, cqs, d2));
  }

 private:
  const std::vector<Eigen::Vector2d> ups_;
  const std::vector<Eigen::Matrix2d> cps_;
  const std::vector<Eigen::Vector2d> uqs_;
  const std::vector<Eigen::Matrix2d> cqs_;
  const double d2_;
};

class D2DNDTCostFunctor2 {
 public:
  D2DNDTCostFunctor2(const std::vector<Eigen::Vector2d> &ups,
                     const std::vector<Eigen::Matrix2d> &cps,
                     const std::vector<Eigen::Vector2d> &uqs,
                     const std::vector<Eigen::Matrix2d> &cqs,
                     double cell_size,
                     double outlier_ratio)
      : ups_(ups),
        cps_(cps),
        uqs_(uqs),
        cqs_(cqs),
        cell_size_(cell_size),
        outlier_ratio_(outlier_ratio) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);
    T c2 = T(outlier_ratio_) / ceres::pow(cell_size_, 2);

    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<T, 2, 1> up = R * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 2> cp = R * cps_[i].cast<T>() * R.transpose();
      Eigen::Matrix<T, 2, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> m = up - uq;
      Eigen::Matrix<T, 2, 2> c = cp + cq;
      T c1 = (1. - outlier_ratio_) / (2 * M_PI * c.determinant());
      T d3 = -ceres::log(c2);
      T d1 = -ceres::log(c1 + c2) - d3;
      T d2 = -T(2) *
             ceres::log((-ceres::log(c1 * ceres::exp(-0.5) + c2) - d3) / d1);
      e[0] += d1 * ceres::exp(-0.5 * d2 * m.dot(c.inverse() * m));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs,
      double cell_size,
      double outlier_ratio) {
    return new ceres::AutoDiffFirstOrderFunction<D2DNDTCostFunctor2, 3>(
        new D2DNDTCostFunctor2(ups, cps, uqs, cqs, cell_size, outlier_ratio));
  }

 private:
  const std::vector<Eigen::Vector2d> ups_;
  const std::vector<Eigen::Matrix2d> cps_;
  const std::vector<Eigen::Vector2d> uqs_;
  const std::vector<Eigen::Matrix2d> cqs_;
  const double cell_size_;
  const double outlier_ratio_;
};

class SNDTCostFunctor2 {
 public:
  SNDTCostFunctor2(const std::vector<Eigen::Vector2d> &ups,
                   const std::vector<Eigen::Matrix2d> &cps,
                   const std::vector<Eigen::Vector2d> &unps,
                   const std::vector<Eigen::Vector2d> &uqs,
                   const std::vector<Eigen::Matrix2d> &cqs,
                   const std::vector<Eigen::Vector2d> &unqs,
                   double d2)
      : ups_(ups),
        unps_(unps),
        uqs_(uqs),
        unqs_(unqs),
        cps_(cps),
        cqs_(cqs),
        d2_(d2) {}

  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<T, 2, 1> up = R * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 1> unp = R * unps_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cp = R * cps_[i].cast<T>() * R.transpose();
      Eigen::Matrix<T, 2, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> unq = unqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();

      Eigen::Matrix<T, 2, 1> m1 = up - uq;
      Eigen::Matrix<T, 2, 1> m2 = (unp + unq).normalized();
      Eigen::Matrix<T, 2, 2> c1 = cp + cq;

      e[0] += -ceres::exp(-d2_ * m1.dot(m2) * m1.dot(m2) / m2.dot(c1 * m2));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &unps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs,
      const std::vector<Eigen::Vector2d> &unqs,
      double d2) {
    return new ceres::AutoDiffFirstOrderFunction<SNDTCostFunctor2, 3>(
        new SNDTCostFunctor2(ups, cps, unps, uqs, cqs, unqs, d2));
  }

 private:
  const std::vector<Eigen::Vector2d> ups_, unps_, uqs_, unqs_;
  const std::vector<Eigen::Matrix2d> cps_, cqs_;
  const double d2_;
};

#include <sndt/cost_functors_arc.h>
