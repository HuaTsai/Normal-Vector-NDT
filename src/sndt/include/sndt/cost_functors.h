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

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &q,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = (T * p - q).norm();
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d p_, q_;
};

class Pt2plICPCostFunctor {
 public:
  Pt2plICPCostFunctor(const Eigen::Vector2d &p,
                      const Eigen::Vector2d &q,
                      const Eigen::Vector2d &nq)
      : p_(p), q_(q), nq_(nq) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> q = q_.cast<T>();
    Eigen::Matrix<T, 2, 1> nq = nq_.cast<T>();

    *e = (p - q).dot(nq);  // Unit is meter since |nq| = 1
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &q,
                                     const Eigen::Vector2d &nq) {
    return new ceres::AutoDiffCostFunction<Pt2plICPCostFunctor, 1, 3>(
        new Pt2plICPCostFunctor(p, q, nq));
  }

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &q,
                     const Eigen::Vector2d &nq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = (T * p - q).dot(nq);
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d p_, q_, nq_;
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

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &np,
                     const Eigen::Vector2d &q,
                     const Eigen::Vector2d &nq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = (T * p - q).dot((T.rotation() * np + nq).normalized());
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d p_, np_, q_, nq_;
};

class D2DNDTCostFunctor {
 public:
  D2DNDTCostFunctor(double d2,
                    const std::vector<Eigen::Vector2d> &ups,
                    const std::vector<Eigen::Matrix2d> &cps,
                    const std::vector<Eigen::Vector2d> &uqs,
                    const std::vector<Eigen::Matrix2d> &cqs)
      : d2_(d2), ups_(ups), cps_(cps), uqs_(uqs), cqs_(cqs), n_(ups.size()) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    for (int i = 0; i < n_; ++i) {
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
      double d2,
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs) {
    return new ceres::AutoDiffFirstOrderFunction<D2DNDTCostFunctor, 3>(
        new D2DNDTCostFunctor(d2, ups, cps, uqs, cqs));
  }

  static double Cost(double d2,
                     const std::vector<Eigen::Vector2d> &ups,
                     const std::vector<Eigen::Matrix2d> &cps,
                     const std::vector<Eigen::Vector2d> &uqs,
                     const std::vector<Eigen::Matrix2d> &cqs,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = 0;
    Eigen::Matrix2d R = T.rotation();
    for (size_t i = 0; i < ups.size(); ++i) {
      Eigen::Vector2d pq = T * ups[i] - uqs[i];
      Eigen::Matrix2d cinv = (R * cps[i] * R.transpose() + cqs[i]).inverse();
      cost += -ceres::exp(-d2 * (pq.dot(cinv * pq)));
    }
    return cost;
  }

 private:
  const double d2_;
  const std::vector<Eigen::Vector2d> ups_;
  const std::vector<Eigen::Matrix2d> cps_;
  const std::vector<Eigen::Vector2d> uqs_;
  const std::vector<Eigen::Matrix2d> cqs_;
  const int n_;
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
      // std::cout << d1 << d2 << std::endl;
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
  SNDTCostFunctor2(double d2,
                   const std::vector<Eigen::Vector2d> &ups,
                   const std::vector<Eigen::Matrix2d> &cps,
                   const std::vector<Eigen::Vector2d> &unps,
                   const std::vector<Eigen::Vector2d> &uqs,
                   const std::vector<Eigen::Matrix2d> &cqs,
                   const std::vector<Eigen::Vector2d> &unqs)
      : d2_(d2),
        ups_(ups),
        unps_(unps),
        uqs_(uqs),
        unqs_(unqs),
        cps_(cps),
        cqs_(cqs),
        n_(ups.size()) {}

  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    for (int i = 0; i < n_; ++i) {
      Eigen::Matrix<T, 2, 1> up = R * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 1> unp = R * unps_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cp = R * cps_[i].cast<T>() * R.transpose();
      Eigen::Matrix<T, 2, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> unq = unqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();

      Eigen::Matrix<T, 2, 1> m1 = up - uq;
      Eigen::Matrix<T, 2, 1> m2 = (unp + unq).normalized();
      Eigen::Matrix<T, 2, 2> c1 = cp + cq;

      T num = m1.dot(m2);
      T den = sqrt(m2.dot(c1 * m2));
      *e += -ceres::exp(-d2_ * ceres::pow(num / den, 2));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      double d2,
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &unps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs,
      const std::vector<Eigen::Vector2d> &unqs) {
    return new ceres::AutoDiffFirstOrderFunction<SNDTCostFunctor2, 3>(
        new SNDTCostFunctor2(d2, ups, cps, unps, uqs, cqs, unqs));
  }

  static double Cost(double d2,
                     const std::vector<Eigen::Vector2d> &ups,
                     const std::vector<Eigen::Matrix2d> &cps,
                     const std::vector<Eigen::Vector2d> &unps,
                     const std::vector<Eigen::Vector2d> &uqs,
                     const std::vector<Eigen::Matrix2d> &cqs,
                     const std::vector<Eigen::Vector2d> &unqs,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = 0;
    for (size_t i = 0; i < ups.size(); ++i) {
      Eigen::Matrix2d R = T.rotation();
      Eigen::Vector2d m1 = T * ups[i] - uqs[i];
      Eigen::Vector2d m2 = (R * unps[i] + unqs[i]).normalized();
      Eigen::Matrix2d c1 = R * cps[i] * R.transpose() + cqs[i];
      cost += -ceres::exp(-d2 * m1.dot(m2) * m1.dot(m2) / m2.dot(c1 * m2));
    }
    return cost;
  }

 private:
  const double d2_;
  const std::vector<Eigen::Vector2d> ups_, unps_, uqs_, unqs_;
  const std::vector<Eigen::Matrix2d> cps_, cqs_;
  const int n_;
};

// class SNDTCostFunctor3 {
//  public:
//   SNDTCostFunctor3(const std::vector<Eigen::Vector2d> &ups,
//                    const std::vector<Eigen::Matrix2d> &cps,
//                    const std::vector<Eigen::Vector2d> &unps,
//                    const std::vector<Eigen::Vector2d> &uqs,
//                    const std::vector<Eigen::Matrix2d> &cqs,
//                    const std::vector<Eigen::Vector2d> &unqs,
//                    double cell_size,
//                    double outlier_ratio)
//       : ups_(ups),
//         unps_(unps),
//         uqs_(uqs),
//         unqs_(unqs),
//         cps_(cps),
//         cqs_(cqs),
//         cell_size_(cell_size),
//         outlier_ratio_(outlier_ratio) {}

//   template <typename T>
//   bool operator()(const T *const xyt, T *e) const {
//     e[0] = T(0);
//     Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
//     Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);
//     T c2 = T(outlier_ratio_) / ceres::pow(cell_size_, 1);

//     for (int i = 0; i < n_; ++i) {
//       Eigen::Matrix<T, 2, 1> up = R * ups_[i].cast<T>() + t;
//       Eigen::Matrix<T, 2, 1> unp = R * unps_[i].cast<T>();
//       Eigen::Matrix<T, 2, 2> cp = R * cps_[i].cast<T>() * R.transpose();
//       Eigen::Matrix<T, 2, 1> uq = uqs_[i].cast<T>();
//       Eigen::Matrix<T, 2, 1> unq = unqs_[i].cast<T>();
//       Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();

//       Eigen::Matrix<T, 2, 1> m1 = up - uq;
//       Eigen::Matrix<T, 2, 1> m2 = (unp + unq).normalized();
//       Eigen::Matrix<T, 2, 2> c1 = cp + cq;

//       T num = m1.dot(m2);
//       T den = sqrt(m2.dot(c1 * m2));
//       T c1 = (1. - outlier_ratio_) / (ceres::sqrt(2 * M_PI * c.determinant()));
//       T d3 = -ceres::log(c2);
//       T d1 = -ceres::log(c1 + c2) - d3;
//       T d2 = -T(2) *
//              ceres::log((-ceres::log(c1 * ceres::exp(-0.5) + c2) - d3) / d1);
//       *e += -ceres::exp(-d2_ * ceres::pow(num / den, 2));
//     }
//     return true;
//   }

//   static ceres::FirstOrderFunction *Create(
//       const std::vector<Eigen::Vector2d> &ups,
//       const std::vector<Eigen::Matrix2d> &cps,
//       const std::vector<Eigen::Vector2d> &unps,
//       const std::vector<Eigen::Vector2d> &uqs,
//       const std::vector<Eigen::Matrix2d> &cqs,
//       const std::vector<Eigen::Vector2d> &unqs,
//       double cell_size,
//       double outlier_ratio) {
//     return new ceres::AutoDiffFirstOrderFunction<SNDTCostFunctor3, 3>(
//         new SNDTCostFunctor3(ups, cps, unps, uqs, cqs, unqs, cell_size,
//                              outlier_ratio));
//   }

//  private:
//   const std::vector<Eigen::Vector2d> ups_, unps_, uqs_, unqs_;
//   const std::vector<Eigen::Matrix2d> cps_, cqs_;
//   const double cell_size_;
//   const double outlier_ratio_;
// };

#include <sndt/cost_functors_arc.h>
