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

class NDTCost2D {
 public:
  NDTCost2D(const std::vector<Eigen::Vector2d> &ups,
            const std::vector<Eigen::Matrix2d> &cps,
            const std::vector<Eigen::Vector2d> &uqs,
            const std::vector<Eigen::Matrix2d> &cqs,
            double d2)
      : ups_(ups), cps_(cps), uqs_(uqs), cqs_(cqs), d2_(d2) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Rotation2D<T> R(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);
    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<T, 2, 1> up = R * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 2> cp = R * cps_[i].cast<T>() * R.inverse();
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
    return new ceres::AutoDiffFirstOrderFunction<NDTCost2D, 3>(
        new NDTCost2D(ups, cps, uqs, cqs, d2));
  }

 private:
  const std::vector<Eigen::Vector2d> ups_;
  const std::vector<Eigen::Matrix2d> cps_;
  const std::vector<Eigen::Vector2d> uqs_;
  const std::vector<Eigen::Matrix2d> cqs_;
  const double d2_;
};

class NNDTCost2D {
 public:
  NNDTCost2D(const std::vector<Eigen::Vector2d> &ups,
             const std::vector<Eigen::Matrix2d> &cps,
             const std::vector<Eigen::Vector2d> &nps,
             const std::vector<Eigen::Vector2d> &uqs,
             const std::vector<Eigen::Matrix2d> &cqs,
             const std::vector<Eigen::Vector2d> &nqs,
             double d2)
      : ups_(ups),
        cps_(cps),
        nps_(nps),
        uqs_(uqs),
        cqs_(cqs),
        nqs_(nqs),
        d2_(d2) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Rotation2D<T> R(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);
    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<T, 2, 1> up = R * ups_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 2> cp = R * cps_[i].cast<T>() * R.inverse();
      Eigen::Matrix<T, 2, 1> np = R * nps_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> nq = nqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> m1 = up - uq;
      Eigen::Matrix<T, 2, 1> m2 = (np + nq).normalized();
      Eigen::Matrix<T, 2, 2> c = cp + cq;
      e[0] += -ceres::exp(-d2_ * m1.dot(m2) * m1.dot(m2) / m2.dot(c * m2));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &nps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs,
      const std::vector<Eigen::Vector2d> &nqs,
      double d2) {
    return new ceres::AutoDiffFirstOrderFunction<NNDTCost2D, 3>(
        new NNDTCost2D(ups, cps, nps, uqs, cqs, nqs, d2));
  }

 private:
  const std::vector<Eigen::Vector2d> ups_;
  const std::vector<Eigen::Matrix2d> cps_;
  const std::vector<Eigen::Vector2d> nps_;
  const std::vector<Eigen::Vector2d> uqs_;
  const std::vector<Eigen::Matrix2d> cqs_;
  const std::vector<Eigen::Vector2d> nqs_;
  const double d2_;
};

namespace {
inline Eigen::Matrix3d RotationMatrix(
    double sx, double sy, double sz, double cx, double cy, double cz) {
  Eigen::Matrix3d ret;
  ret.row(0) << cy * cz, -sz * cy, sy;
  ret.row(1) << sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy;
  ret.row(2) << sx * sz - sy * cx * cz, sx * cz + sy * sz * cx, cx * cy;
  return ret;
}

inline Eigen::Matrix<double, 9, 3> DerivativeOfRotation(
    double sx, double sy, double sz, double cx, double cy, double cz) {
  Eigen::Matrix<double, 9, 3> ret;
  ret.row(0).setZero();
  ret.row(1) << -sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy;
  ret.row(2) << sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy;
  ret.row(3) << -sy * cz, sy * sz, cy;
  ret.row(4) << sx * cy * cz, -sx * sz * cy, sx * sy;
  ret.row(5) << -cx * cy * cz, sz * cx * cy, -sy * cx;
  ret.row(6) << -sz * cy, -cy * cz, 0.;
  ret.row(7) << -sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.;
  ret.row(8) << sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.;
  return ret;
}
}  // namespace

class NDTCostN final : public ceres::FirstOrderFunction {
 public:
  NDTCostN(const std::vector<Eigen::Vector3d> &ups,
           const std::vector<Eigen::Matrix3d> &cps,
           const std::vector<Eigen::Vector3d> &uqs,
           const std::vector<Eigen::Matrix3d> &cqs,
           double d2)
      : ups_(ups), cps_(cps), uqs_(uqs), cqs_(cqs), d2_(d2) {}

  bool Evaluate(const double *const p, double *f, double *g) const override {
    f[0] = 0;
    memset(g, 0, sizeof(double) * 6);

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix<double, 9, 3> dR;
    dc(p, R, t, dR);

    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<double, 3, 6> jupq;
      Eigen::Matrix<double, 3, 18> jSpq;
      dp(ups_[i], cps_[i], R, dR, jupq, jSpq);
      Eigen::Matrix3d B = (R * cps_[i] * R.transpose() + cqs_[i]).inverse();
      Eigen::Vector3d upq = R * ups_[i] + t - uqs_[i];
      Eigen::Transpose<Eigen::Vector3d> upqT(upq);
      double expval = std::exp(-0.5 * d2_ * upqT * B * upq);
      f[0] -= expval;
      for (int a = 0; a < 6; ++a) {
        Eigen::Ref<Eigen::Vector3d> ja(jupq.block<3, 1>(0, a));
        Eigen::Ref<Eigen::Matrix3d> Za(jSpq.block<3, 3>(0, 3 * a));
        double qa = (2 * upqT * B * ja - upqT * B * Za * B * upq)(0);
        g[a] += 0.5 * d2_ * expval * qa;
      }
    }
    return true;
  }

  int NumParameters() const override { return 6; }

  void dp(const Eigen::Vector3d &mean,
          const Eigen::Matrix3d &cov,
          const Eigen::Matrix3d &R,
          const Eigen::Matrix<double, 9, 3> &dR,
          Eigen::Matrix<double, 3, 6> &jupq,
          Eigen::Matrix<double, 3, 18> &jSpq) const {
    Eigen::Matrix<double, 9, 1> j = dR * mean;
    jupq.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jupq.col(3) = j.segment(0, 3);
    jupq.col(4) = j.segment(3, 6);
    jupq.col(5) = j.segment(6, 9);

    Eigen::Matrix<double, 9, 3> c = dR * cov * R.transpose();
    // clang-format off
    jSpq.setZero();
    jSpq.block<3, 3>(0, 9) = c.block<3, 3>(0, 0) + c.block<3, 3>(0, 0).transpose();
    jSpq.block<3, 3>(0, 12) = c.block<3, 3>(3, 0) + c.block<3, 3>(3, 0).transpose();
    jSpq.block<3, 3>(0, 15) = c.block<3, 3>(6, 0) + c.block<3, 3>(6, 0).transpose();
    // clang-format on
  }

  void dc(const double *const p,
          Eigen::Matrix3d &R,
          Eigen::Vector3d &t,
          Eigen::Matrix<double, 9, 3> &dR) const {
    double sx = std::sin(p[3]), cx = std::cos(p[3]);
    double sy = std::sin(p[4]), cy = std::cos(p[4]);
    double sz = std::sin(p[5]), cz = std::cos(p[5]);
    R = RotationMatrix(sx, sy, sz, cx, cy, cz);
    t << p[0], p[1], p[2];
    dR = DerivativeOfRotation(sx, sy, sz, cx, cy, cz);
  }

 private:
  const std::vector<Eigen::Vector3d> ups_;
  const std::vector<Eigen::Matrix3d> cps_;
  const std::vector<Eigen::Vector3d> uqs_;
  const std::vector<Eigen::Matrix3d> cqs_;
  const double d2_;
};

class NNDTCostN final : public ceres::FirstOrderFunction {
 public:
  NNDTCostN(const std::vector<Eigen::Vector3d> &ups,
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

  bool Evaluate(const double *const p, double *f, double *g) const override {
    f[0] = 0;
    memset(g, 0, sizeof(double) * 6);

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix<double, 9, 3> dR;
    dc(p, R, t, dR);

    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<double, 3, 6> jupq;
      Eigen::Matrix<double, 3, 18> jSpq;
      Eigen::Matrix<double, 3, 6> jnpq;
      dp(ups_[i], cps_[i], nps_[i], R, dR, jupq, jSpq, jnpq);
      Eigen::Vector3d upq = R * ups_[i] + t - uqs_[i];
      Eigen::Matrix3d Spq = R * cps_[i] * R.transpose() + cqs_[i];
      Eigen::Vector3d npq = R * nps_[i] + nqs_[i];
      Eigen::Transpose<Eigen::Vector3d> upqT(upq);
      Eigen::Transpose<Eigen::Vector3d> npqT(npq);
      double num = std::pow((upqT * npq)(0), 2);
      double den = (npqT * Spq * npq)(0);
      double expval = std::exp(-0.5 * d2_ * num / den);
      f[0] -= expval;
      double factor = 0.5 * d2_ * expval * std::pow(den, -2);
      for (int a = 0; a < 6; ++a) {
        Eigen::Ref<Eigen::Vector3d> jua(jupq.block<3, 1>(0, a));
        Eigen::Ref<Eigen::Matrix3d> jSa(jSpq.block<3, 3>(0, 3 * a));
        Eigen::Ref<Eigen::Vector3d> jna(jnpq.block<3, 1>(0, a));
        double dnum = 2 * (upqT * npq)(0) * (npqT * jua + upqT * jna)(0);
        double dden = (2 * npqT * Spq * jna + npqT * jSa * npq)(0);
        g[a] += factor * (dnum * den - num * dden);
      }
    }
    return true;
  }

  int NumParameters() const override { return 6; }

  void dp(const Eigen::Vector3d &mean,
          const Eigen::Matrix3d &cov,
          const Eigen::Vector3d &normal,
          const Eigen::Matrix3d &R,
          const Eigen::Matrix<double, 9, 3> &dR,
          Eigen::Matrix<double, 3, 6> &jupq,
          Eigen::Matrix<double, 3, 18> &jSpq,
          Eigen::Matrix<double, 3, 6> &jnpq) const {
    Eigen::Matrix<double, 9, 1> j = dR * mean;
    jupq.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jupq.col(3) = j.segment(0, 3);
    jupq.col(4) = j.segment(3, 6);
    jupq.col(5) = j.segment(6, 9);

    Eigen::Matrix<double, 9, 1> n = dR * normal;
    jnpq.block<3, 3>(0, 0) = Eigen::Matrix3d::Zero();
    jnpq.col(3) = n.segment(0, 3);
    jnpq.col(4) = n.segment(3, 6);
    jnpq.col(5) = n.segment(6, 9);

    Eigen::Matrix<double, 9, 3> c = dR * cov * R.transpose();
    // clang-format off
    jSpq.setZero();
    jSpq.block<3, 3>(0, 9) = c.block<3, 3>(0, 0) + c.block<3, 3>(0, 0).transpose();
    jSpq.block<3, 3>(0, 12) = c.block<3, 3>(3, 0) + c.block<3, 3>(3, 0).transpose();
    jSpq.block<3, 3>(0, 15) = c.block<3, 3>(6, 0) + c.block<3, 3>(6, 0).transpose();
    // clang-format on
  }

  void dc(const double *const p,
          Eigen::Matrix3d &R,
          Eigen::Vector3d &t,
          Eigen::Matrix<double, 9, 3> &dR) const {
    double sx = std::sin(p[3]), cx = std::cos(p[3]);
    double sy = std::sin(p[4]), cy = std::cos(p[4]);
    double sz = std::sin(p[5]), cz = std::cos(p[5]);
    R = RotationMatrix(sx, sy, sz, cx, cy, cz);
    t << p[0], p[1], p[2];
    dR = DerivativeOfRotation(sx, sy, sz, cx, cy, cz);
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

class ICPCost final : public ceres::FirstOrderFunction {
 public:
  ICPCost(const std::vector<Eigen::Vector3d> &ps,
          const std::vector<Eigen::Vector3d> &qs)
      : ps_(ps), qs_(qs) {}

  bool Evaluate(const double *const p, double *f, double *g) const override {
    f[0] = 0;
    memset(g, 0, sizeof(double) * 6);

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix<double, 9, 3> dR;
    dc(p, R, t, dR);

    for (size_t i = 0; i < ps_.size(); ++i) {
      Eigen::Matrix<double, 3, 6> jpq;
      Eigen::Matrix<double, 9, 1> j = dR * ps_[i];
      jpq.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      jpq.col(3) = j.segment(0, 3);
      jpq.col(4) = j.segment(3, 6);
      jpq.col(5) = j.segment(6, 9);
      Eigen::Vector3d pq = R * ps_[i] + t - qs_[i];
      f[0] += pq.squaredNorm();
      Eigen::Matrix<double, 6, 1> gvec = jpq.transpose() * pq;
      for (int i = 0; i < 6; ++i) g[i] += 2 * gvec(i);
    }
    return true;
  }

  int NumParameters() const override { return 6; }

  void dc(const double *const p,
          Eigen::Matrix3d &R,
          Eigen::Vector3d &t,
          Eigen::Matrix<double, 9, 3> &dR) const {
    double sx = std::sin(p[3]), cx = std::cos(p[3]);
    double sy = std::sin(p[4]), cy = std::cos(p[4]);
    double sz = std::sin(p[5]), cz = std::cos(p[5]);
    R = RotationMatrix(sx, sy, sz, cx, cy, cz);
    t << p[0], p[1], p[2];
    dR = DerivativeOfRotation(sx, sy, sz, cx, cy, cz);
  }

 private:
  const std::vector<Eigen::Vector3d> ps_;
  const std::vector<Eigen::Vector3d> qs_;
};

class SICPCost final : public ceres::FirstOrderFunction {
 public:
  SICPCost(const std::vector<Eigen::Vector3d> &ps,
           const std::vector<Eigen::Vector3d> &nps,
           const std::vector<Eigen::Vector3d> &qs,
           const std::vector<Eigen::Vector3d> &nqs)
      : ps_(ps), nps_(nps), qs_(qs), nqs_(nqs) {}

  bool Evaluate(const double *const p, double *f, double *g) const override {
    f[0] = 0;
    memset(g, 0, sizeof(double) * 6);

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix<double, 9, 3> dR;
    dc(p, R, t, dR);

    for (size_t i = 0; i < ps_.size(); ++i) {
      Eigen::Matrix<double, 3, 6> jpq;
      Eigen::Matrix<double, 9, 1> j = dR * ps_[i];
      jpq.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      jpq.col(3) = j.segment(0, 3);
      jpq.col(4) = j.segment(3, 6);
      jpq.col(5) = j.segment(6, 9);

      Eigen::Matrix<double, 3, 6> jnpq;
      Eigen::Matrix<double, 9, 1> n = dR * nps_[i];
      jnpq.block<3, 3>(0, 0) = Eigen::Matrix3d::Zero();
      jnpq.col(3) = n.segment(0, 3);
      jnpq.col(4) = n.segment(3, 6);
      jnpq.col(5) = n.segment(6, 9);

      Eigen::Vector3d pq = R * ps_[i] + t - qs_[i];
      Eigen::Vector3d npq = nps_[i] + nqs_[i];
      f[0] += std::pow(pq.dot(npq), 2) / npq.squaredNorm();
      for (int a = 0; a < 6; ++a) {
        Eigen::Ref<Eigen::Vector3d> ja(jpq.block<3, 1>(0, a));
        Eigen::Ref<Eigen::Vector3d> jna(jnpq.block<3, 1>(0, a));
        double val = (ja.transpose() * npq + pq.transpose() * jna)(0);
        g[a] += 2 * pq.dot(npq) * val / npq.squaredNorm();
      }
    }
    return true;
  }

  int NumParameters() const override { return 6; }

  void dc(const double *const p,
          Eigen::Matrix3d &R,
          Eigen::Vector3d &t,
          Eigen::Matrix<double, 9, 3> &dR) const {
    double sx = std::sin(p[3]), cx = std::cos(p[3]);
    double sy = std::sin(p[4]), cy = std::cos(p[4]);
    double sz = std::sin(p[5]), cz = std::cos(p[5]);
    R = RotationMatrix(sx, sy, sz, cx, cy, cz);
    t << p[0], p[1], p[2];
    dR = DerivativeOfRotation(sx, sy, sz, cx, cy, cz);
  }

 private:
  const std::vector<Eigen::Vector3d> ps_;
  const std::vector<Eigen::Vector3d> nps_;
  const std::vector<Eigen::Vector3d> qs_;
  const std::vector<Eigen::Vector3d> nqs_;
};
