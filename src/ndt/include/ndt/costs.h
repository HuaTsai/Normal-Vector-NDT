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

class CostObj {
 public:
  virtual ~CostObj() = default;
  virtual double operator()(const Eigen::VectorXd &x,
                            Eigen::VectorXd &grad) = 0;
};

class NDTCostObj : public CostObj {
 public:
  NDTCostObj(const std::vector<Eigen::Vector3d> &ups,
             const std::vector<Eigen::Matrix3d> &cps,
             const std::vector<Eigen::Vector3d> &uqs,
             const std::vector<Eigen::Matrix3d> &cqs,
             double d2)
      : ups_(ups), cps_(cps), uqs_(uqs), cqs_(cqs), d2_(d2) {}

  double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override {
    double f = 0;
    grad.setZero();
    dc(x);
    for (size_t i = 0; i < ups_.size(); ++i) {
      dp(ups_[i], cps_[i]);
      Eigen::Matrix3d B = (R_ * cps_[i] * R_.transpose() + cqs_[i]).inverse();
      Eigen::Vector3d uij = R_ * ups_[i] + t_ - uqs_[i];
      Eigen::Transpose<Eigen::Vector3d> uijT(uij);
      double expval = std::exp(-0.5 * d2_ * uijT * B * uij);
      f -= expval;
      for (int a = 0; a < 6; ++a) {
        const Eigen::Ref<Eigen::Vector3d> ja(J_.block<3, 1>(0, a));
        const Eigen::Ref<Eigen::Matrix3d> Za(C_.block<3, 3>(0, 3 * a));
        double qa = (2 * uijT * B * ja - uijT * B * Za * B * uij)(0);
        grad(a) += 0.5 * d2_ * expval * qa;
      }
    }
    return f;
  }

  void dp(const Eigen::Vector3d &mean, const Eigen::Matrix3d &cov) {
    Eigen::Matrix<double, 8, 1> j = j_com_ * mean;
    J_.col(0) = Eigen::Vector3d::UnitX();
    J_.col(1) = Eigen::Vector3d::UnitY();
    J_.col(2) = Eigen::Vector3d::UnitZ();
    J_.col(3) = Eigen::Vector3d(0, j(0), j(1));
    J_.col(4) = Eigen::Vector3d(j(2), j(3), j(4));
    J_.col(5) = Eigen::Vector3d(j(5), j(6), j(7));

    Eigen::Matrix<double, 9, 3> c = c_com_ * cov * R_.transpose();
    // clang-format off
    C_.setZero();
    C_.block<3, 3>(0, 9) = c.block<3, 3>(0, 0) + c.block<3, 3>(0, 0).transpose();
    C_.block<3, 3>(0, 12) = c.block<3, 3>(3, 0) + c.block<3, 3>(3, 0).transpose();
    C_.block<3, 3>(0, 15) = c.block<3, 3>(6, 0) + c.block<3, 3>(6, 0).transpose();
    // clang-format on
  }

  void dc(const Eigen::Matrix<double, 6, 1> &p) {
    const auto cal = [](double angle, double &c, double &s) {
      c = std::cos(angle), s = std::sin(angle);
    };
    double cx, cy, cz, sx, sy, sz;
    cal(p(3), cx, sx);
    cal(p(4), cy, sy);
    cal(p(5), cz, sz);

    // clang-format off
    R_.row(0) = Eigen::Vector3d(cy * cz, -sz * cy, sy);
    R_.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    R_.row(2) = Eigen::Vector3d(sx * sz - sy * cx * cz, sx * cz + sy * sz * cx, cx * cy);
    t_ = p.head(3);

    j_com_.row(0) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    j_com_.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    j_com_.row(2) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    j_com_.row(3) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    j_com_.row(4) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    j_com_.row(5) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    j_com_.row(6) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    j_com_.row(7) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);

    c_com_.row(0) = Eigen::Vector3d::Zero();
    c_com_.row(1) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    c_com_.row(2) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    c_com_.row(3) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    c_com_.row(4) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    c_com_.row(5) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    c_com_.row(6) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    c_com_.row(7) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    c_com_.row(8) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);
    // clang-format on
  }

 private:
  const std::vector<Eigen::Vector3d> ups_;
  const std::vector<Eigen::Matrix3d> cps_;
  const std::vector<Eigen::Vector3d> uqs_;
  const std::vector<Eigen::Matrix3d> cqs_;
  const double d2_;

  Eigen::Matrix<double, 8, 3> j_com_;
  Eigen::Matrix<double, 9, 3> c_com_;
  Eigen::Matrix<double, 3, 6> J_;
  Eigen::Matrix<double, 3, 18> C_;
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
};

class NNDTCostObj : public CostObj {
 public:
  NNDTCostObj(const std::vector<Eigen::Vector3d> &ups,
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

  double operator()(const Eigen::VectorXd &p, Eigen::VectorXd &grad) override {
    double f = 0;
    grad.setZero();
    dc(p);
    for (size_t i = 0; i < ups_.size(); ++i) {
      dp(ups_[i], cps_[i], nps_[i]);
      Eigen::Vector3d upq = R_ * ups_[i] + t_ - uqs_[i];
      Eigen::Matrix3d Spq = R_ * cps_[i] * R_.transpose() + cqs_[i];
      Eigen::Vector3d npq = R_ * nps_[i] + nqs_[i];
      Eigen::Transpose<Eigen::Vector3d> upqT(upq);
      Eigen::Transpose<Eigen::Vector3d> npqT(npq);
      double num = std::pow((upqT * npq)(0), 2);
      double den = (npqT * Spq * npq)(0);
      double expval = std::exp(-0.5 * d2_ * num / den);
      f -= expval;
      double factor = 0.5 * d2_ * expval * std::pow(den, -2);
      for (int a = 0; a < 6; ++a) {
        const Eigen::Ref<Eigen::Vector3d> jua(jupq_.block<3, 1>(0, a));
        const Eigen::Ref<Eigen::Matrix3d> jSa(jSpq_.block<3, 3>(0, 3 * a));
        const Eigen::Ref<Eigen::Vector3d> jna(jnpq_.block<3, 1>(0, a));
        double dnum = 2 * (upqT * npq)(0) * (npqT * jua + upqT * jna)(0);
        double dden = (2 * npqT * Spq * jna + npqT * jSa * npq)(0);
        grad(a) += factor * (dnum * den - num * dden);
      }
    }
    return f;
  }

  void dp(const Eigen::Vector3d &mean,
          const Eigen::Matrix3d &cov,
          const Eigen::Vector3d &normal) {
    Eigen::Matrix<double, 8, 1> j = jcom_ * mean;
    jupq_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jupq_.col(3) = Eigen::Vector3d(0, j(0), j(1));
    jupq_.col(4) = Eigen::Vector3d(j(2), j(3), j(4));
    jupq_.col(5) = Eigen::Vector3d(j(5), j(6), j(7));

    Eigen::Matrix<double, 8, 1> j2 = jcom_ * normal;
    jnpq_.block<3, 3>(0, 0) = Eigen::Matrix3d::Zero();
    jnpq_.col(3) = Eigen::Vector3d(0, j2(0), j2(1));
    jnpq_.col(4) = Eigen::Vector3d(j2(2), j2(3), j2(4));
    jnpq_.col(5) = Eigen::Vector3d(j2(5), j2(6), j2(7));

    Eigen::Matrix<double, 9, 3> c = ccom_ * cov * R_.transpose();
    // clang-format off
    jSpq_.setZero();
    jSpq_.block<3, 3>(0, 9) = c.block<3, 3>(0, 0) + c.block<3, 3>(0, 0).transpose();
    jSpq_.block<3, 3>(0, 12) = c.block<3, 3>(3, 0) + c.block<3, 3>(3, 0).transpose();
    jSpq_.block<3, 3>(0, 15) = c.block<3, 3>(6, 0) + c.block<3, 3>(6, 0).transpose();
    // clang-format on
  }

  void dc(const Eigen::Matrix<double, 6, 1> &p) {
    const auto cal = [](double angle, double &c, double &s) {
      c = std::cos(angle), s = std::sin(angle);
    };
    double cx, cy, cz, sx, sy, sz;
    cal(p(3), cx, sx);
    cal(p(4), cy, sy);
    cal(p(5), cz, sz);

    // clang-format off
    R_.row(0) = Eigen::Vector3d(cy * cz, -sz * cy, sy);
    R_.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    R_.row(2) = Eigen::Vector3d(sx * sz - sy * cx * cz, sx * cz + sy * sz * cx, cx * cy);
    t_ = p.head(3);

    jcom_.row(0) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    jcom_.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    jcom_.row(2) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    jcom_.row(3) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    jcom_.row(4) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    jcom_.row(5) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    jcom_.row(6) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    jcom_.row(7) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);

    ccom_.row(0) = Eigen::Vector3d::Zero();
    ccom_.row(1) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    ccom_.row(2) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    ccom_.row(3) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    ccom_.row(4) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    ccom_.row(5) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    ccom_.row(6) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    ccom_.row(7) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    ccom_.row(8) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);
    // clang-format on
  }

 private:
  const std::vector<Eigen::Vector3d> ups_;
  const std::vector<Eigen::Matrix3d> cps_;
  const std::vector<Eigen::Vector3d> nps_;
  const std::vector<Eigen::Vector3d> uqs_;
  const std::vector<Eigen::Matrix3d> cqs_;
  const std::vector<Eigen::Vector3d> nqs_;
  const double d2_;

  Eigen::Matrix<double, 8, 3> jcom_;
  Eigen::Matrix<double, 9, 3> ccom_;
  Eigen::Matrix<double, 3, 6> jupq_;
  Eigen::Matrix<double, 3, 6> jnpq_;
  Eigen::Matrix<double, 3, 18> jSpq_;
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
};

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
    Eigen::Matrix<double, 8, 3> jcom;
    Eigen::Matrix<double, 9, 3> ccom;

    dc(p, R, t, jcom, ccom);
    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<double, 3, 6> jupq;
      Eigen::Matrix<double, 3, 18> jSpq;
      dp(ups_[i], cps_[i], R, jcom, ccom, jupq, jSpq);
      Eigen::Matrix3d B = (R * cps_[i] * R.transpose() + cqs_[i]).inverse();
      Eigen::Vector3d upq = R * ups_[i] + t - uqs_[i];
      Eigen::Transpose<Eigen::Vector3d> upqT(upq);
      double expval = std::exp(-0.5 * d2_ * upqT * B * upq);
      f[0] -= expval;
      for (int a = 0; a < 6; ++a) {
        const Eigen::Ref<Eigen::Vector3d> ja(jupq.block<3, 1>(0, a));
        const Eigen::Ref<Eigen::Matrix3d> Za(jSpq.block<3, 3>(0, 3 * a));
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
          const Eigen::Matrix<double, 8, 3> &jcom,
          const Eigen::Matrix<double, 9, 3> &ccom,
          Eigen::Matrix<double, 3, 6> &jupq,
          Eigen::Matrix<double, 3, 18> &jSpq) const {
    Eigen::Matrix<double, 8, 1> j = jcom * mean;
    jupq.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jupq.col(3) = Eigen::Vector3d(0, j(0), j(1));
    jupq.col(4) = Eigen::Vector3d(j(2), j(3), j(4));
    jupq.col(5) = Eigen::Vector3d(j(5), j(6), j(7));

    Eigen::Matrix<double, 9, 3> c = ccom * cov * R.transpose();
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
          Eigen::Matrix<double, 8, 3> &jcom,
          Eigen::Matrix<double, 9, 3> &ccom) const {
    const auto cal = [](double angle, double &c, double &s) {
      c = std::cos(angle), s = std::sin(angle);
    };
    double cx, cy, cz, sx, sy, sz;
    cal(p[3], cx, sx);
    cal(p[4], cy, sy);
    cal(p[5], cz, sz);

    // clang-format off
    R.row(0) = Eigen::Vector3d(cy * cz, -sz * cy, sy);
    R.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    R.row(2) = Eigen::Vector3d(sx * sz - sy * cx * cz, sx * cz + sy * sz * cx, cx * cy);
    t = Eigen::Vector3d(p[0], p[1], p[2]);

    jcom.row(0) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    jcom.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    jcom.row(2) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    jcom.row(3) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    jcom.row(4) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    jcom.row(5) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    jcom.row(6) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    jcom.row(7) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);

    ccom.row(0) = Eigen::Vector3d::Zero();
    ccom.row(1) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    ccom.row(2) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    ccom.row(3) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    ccom.row(4) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    ccom.row(5) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    ccom.row(6) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    ccom.row(7) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    ccom.row(8) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);
    // clang-format on
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
    Eigen::Matrix<double, 8, 3> jcom;
    Eigen::Matrix<double, 9, 3> ccom;

    dc(p, R, t, jcom, ccom);
    for (size_t i = 0; i < ups_.size(); ++i) {
      Eigen::Matrix<double, 3, 6> jupq;
      Eigen::Matrix<double, 3, 18> jSpq;
      Eigen::Matrix<double, 3, 6> jnpq;
      dp(ups_[i], cps_[i], nps_[i], R, jcom, ccom, jupq, jSpq, jnpq);
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
        const Eigen::Ref<Eigen::Vector3d> jua(jupq.block<3, 1>(0, a));
        const Eigen::Ref<Eigen::Matrix3d> jSa(jSpq.block<3, 3>(0, 3 * a));
        const Eigen::Ref<Eigen::Vector3d> jna(jnpq.block<3, 1>(0, a));
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
          const Eigen::Matrix<double, 8, 3> &jcom,
          const Eigen::Matrix<double, 9, 3> &ccom,
          Eigen::Matrix<double, 3, 6> &jupq,
          Eigen::Matrix<double, 3, 18> &jSpq,
          Eigen::Matrix<double, 3, 6> &jnpq) const {
    Eigen::Matrix<double, 8, 1> j = jcom * mean;
    jupq.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jupq.col(3) = Eigen::Vector3d(0, j(0), j(1));
    jupq.col(4) = Eigen::Vector3d(j(2), j(3), j(4));
    jupq.col(5) = Eigen::Vector3d(j(5), j(6), j(7));

    Eigen::Matrix<double, 8, 1> j2 = jcom * normal;
    jnpq.block<3, 3>(0, 0) = Eigen::Matrix3d::Zero();
    jnpq.col(3) = Eigen::Vector3d(0, j2(0), j2(1));
    jnpq.col(4) = Eigen::Vector3d(j2(2), j2(3), j2(4));
    jnpq.col(5) = Eigen::Vector3d(j2(5), j2(6), j2(7));

    Eigen::Matrix<double, 9, 3> c = ccom * cov * R.transpose();
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
          Eigen::Matrix<double, 8, 3> &jcom,
          Eigen::Matrix<double, 9, 3> &ccom) const {
    const auto cal = [](double angle, double &c, double &s) {
      c = std::cos(angle), s = std::sin(angle);
    };
    double cx, cy, cz, sx, sy, sz;
    cal(p[3], cx, sx);
    cal(p[4], cy, sy);
    cal(p[5], cz, sz);

    // clang-format off
    R.row(0) = Eigen::Vector3d(cy * cz, -sz * cy, sy);
    R.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    R.row(2) = Eigen::Vector3d(sx * sz - sy * cx * cz, sx * cz + sy * sz * cx, cx * cy);
    t = Eigen::Vector3d(p[0], p[1], p[2]);

    jcom.row(0) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    jcom.row(1) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    jcom.row(2) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    jcom.row(3) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    jcom.row(4) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    jcom.row(5) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    jcom.row(6) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    jcom.row(7) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);

    ccom.row(0) = Eigen::Vector3d::Zero();
    ccom.row(1) = Eigen::Vector3d(-sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy);
    ccom.row(2) = Eigen::Vector3d(sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy);
    ccom.row(3) = Eigen::Vector3d(-sy * cz, sy * sz, cy);
    ccom.row(4) = Eigen::Vector3d(sx * cy * cz, -sx * sz * cy, sx * sy);
    ccom.row(5) = Eigen::Vector3d(-cx * cy * cz, sz * cx * cy, -sy * cx);
    ccom.row(6) = Eigen::Vector3d(-sz * cy, -cy * cz, 0.);
    ccom.row(7) = Eigen::Vector3d(-sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0.);
    ccom.row(8) = Eigen::Vector3d(sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0.);
    // clang-format on
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
