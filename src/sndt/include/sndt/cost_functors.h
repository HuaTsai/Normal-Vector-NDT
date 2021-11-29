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

// Note: It seems that we do not need this feature.
class AngleLocalParameterization {
 public:
  template <typename T>
  bool operator()(const T *t, const T *dt, T *t_add_dt) const {
    T pi(M_PI), twopi(2. * M_PI), sum(*t + *dt);
    *t_add_dt = sum - twopi * ceres::floor((sum + pi) / twopi);
    return true;
  }

  static ceres::LocalParameterization *Create() {
    return new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                    1, 1>;
  }
};

class ICPCostFunctor {
 public:
  ICPCostFunctor(const Eigen::Vector2d &p, const Eigen::Vector2d &q)
      : p_(p), q_(q) {}
  template <typename T>
  bool operator()(const T *const x,
                  const T *const y,
                  const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> q = q_.cast<T>();

    // Note: (p - q).norm() works, but with warning message (nan Jacobian)
    e[0] = (p - q)[0];
    e[1] = (p - q)[1];
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &q) {
    return new ceres::AutoDiffCostFunction<ICPCostFunctor, 2, 1, 1, 1>(
        new ICPCostFunctor(p, q));
  }

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &q,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    return (T * p - q).norm();
  }

  static double Cost2(const Eigen::Vector2d &p,
                      const Eigen::Vector2d &q,
                      const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = Cost(p, q, T);
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
  bool operator()(const T *const x,
                  const T *const y,
                  const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> q = q_.cast<T>();
    Eigen::Matrix<T, 2, 1> nq = nq_.cast<T>();

    *e = (p - q).dot(nq);  // Unit is meter since |nq| = 1
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &q,
                                     const Eigen::Vector2d &nq) {
    return new ceres::AutoDiffCostFunction<Pt2plICPCostFunctor, 1, 1, 1, 1>(
        new Pt2plICPCostFunctor(p, q, nq));
  }

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &q,
                     const Eigen::Vector2d &nq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    return std::abs((T * p - q).dot(nq));
  }

  static double Cost2(const Eigen::Vector2d &p,
                      const Eigen::Vector2d &q,
                      const Eigen::Vector2d &nq,
                      const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = Cost(p, q, nq, T);
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
  bool operator()(const T *const x,
                  const T *const y,
                  const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> np = R * np_.cast<T>();
    Eigen::Matrix<T, 2, 1> q = q_.cast<T>();
    Eigen::Matrix<T, 2, 1> nq = nq_.cast<T>();
    // FIXME: normalize np + nq or not

    *e = (p - q).dot(np + nq);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &np,
                                     const Eigen::Vector2d &q,
                                     const Eigen::Vector2d &nq) {
    return new ceres::AutoDiffCostFunction<SICPCostFunctor, 1, 1, 1, 1>(
        new SICPCostFunctor(p, np, q, nq));
  }

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &np,
                     const Eigen::Vector2d &q,
                     const Eigen::Vector2d &nq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    return std::abs((T * p - q).dot(T.rotation() * np + nq));
  }

  static double Cost2(const Eigen::Vector2d &p,
                      const Eigen::Vector2d &np,
                      const Eigen::Vector2d &q,
                      const Eigen::Vector2d &nq,
                      const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = Cost(p, np, q, nq, T);
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d p_, np_, q_, nq_;
};

class P2DNDTMDCostFunctor {
 public:
  P2DNDTMDCostFunctor(const Eigen::Vector2d &p,
                      const Eigen::Vector2d &uq,
                      const Eigen::Matrix2d &cq)
      : p_(p), uq_(uq), cq_(cq) {}
  template <typename T>
  bool operator()(const T *const x,
                  const T *const y,
                  const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 2, 2> cq = cq_.cast<T>();
    Eigen::Matrix<T, 2, 1> pq = p - uq;

    // TODO: check if .dot( * ) is okay or not
    *e = sqrt((pq.transpose() * cq.inverse() * pq)[0]);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &uq,
                                     const Eigen::Matrix2d &cq) {
    return new ceres::AutoDiffCostFunction<P2DNDTMDCostFunctor, 1, 1, 1, 1>(
        new P2DNDTMDCostFunctor(p, uq, cq));
  }

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &uq,
                     const Eigen::Matrix2d &cq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    Eigen::Vector2d pq = T * p - uq;
    Eigen::Matrix2d cinv = cq.inverse();
    return sqrt(pq.dot(cinv * pq));
  }

  static double Cost2(const Eigen::Vector2d &p,
                      const Eigen::Vector2d &uq,
                      const Eigen::Matrix2d &cq,
                      const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = Cost(p, uq, cq, T);
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d p_, uq_;
  const Eigen::Matrix2d cq_;
};

class P2DNDTCostFunctor {
 public:
  P2DNDTCostFunctor(const std::vector<Eigen::Vector2d> &ps,
                    const std::vector<Eigen::Vector2d> &qs,
                    const std::vector<Eigen::Matrix2d> &cqs)
      : ps_(ps), qs_(qs), cqs_(cqs), n_(ps.size()) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    e[0] = T(0);
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    for (int i = 0; i < n_; ++i) {
      Eigen::Matrix<T, 2, 1> p = R * ps_[i].cast<T>() + t;
      Eigen::Matrix<T, 2, 1> q = qs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> pq = p - q;
      e[0] += -ceres::exp(-0.5 * (pq.transpose() * cq.inverse() * pq)[0]);
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector2d> &ps,
      const std::vector<Eigen::Vector2d> &qs,
      const std::vector<Eigen::Matrix2d> &cqs) {
    return new ceres::AutoDiffFirstOrderFunction<P2DNDTCostFunctor, 3>(
        new P2DNDTCostFunctor(ps, qs, cqs));
  }

  static double Cost(const std::vector<Eigen::Vector2d> &ps,
                     const std::vector<Eigen::Vector2d> &qs,
                     const std::vector<Eigen::Matrix2d> &cqs,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
      Eigen::Vector2d pq = T * ps[i] - qs[i];
      Eigen::Matrix2d cinv = cqs[i].inverse();
      cost += -ceres::exp(-0.5 * (pq.dot(cinv * pq)));
    }
    return cost;
  }

 private:
  const std::vector<Eigen::Vector2d> ps_, qs_;
  const std::vector<Eigen::Matrix2d> cqs_;
  const int n_;
};

class D2DNDTMDCostFunctor {
 public:
  D2DNDTMDCostFunctor(const Eigen::Vector2d &up,
                      const Eigen::Matrix2d &cp,
                      const Eigen::Vector2d &uq,
                      const Eigen::Matrix2d &cq)
      : up_(up), cp_(cp), uq_(uq), cq_(cq) {}
  template <typename T>
  bool operator()(const T *const x,
                  const T *const y,
                  const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up = R * up_.cast<T>() + t;
    Eigen::Matrix<T, 2, 2> cp = R * cp_.cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 2, 2> cq = cq_.cast<T>();
    Eigen::Matrix<T, 2, 1> m = up - uq;
    Eigen::Matrix<T, 2, 2> c = cp + cq;

    *e = sqrt((m.transpose() * c.inverse() * m)[0]);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &up,
                                     const Eigen::Matrix2d &cp,
                                     const Eigen::Vector2d &uq,
                                     const Eigen::Matrix2d &cq) {
    return new ceres::AutoDiffCostFunction<D2DNDTMDCostFunctor, 1, 1, 1, 1>(
        new D2DNDTMDCostFunctor(up, cp, uq, cq));
  }

  static double Cost(const Eigen::Vector2d &up,
                     const Eigen::Matrix2d &cp,
                     const Eigen::Vector2d &uq,
                     const Eigen::Matrix2d &cq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    Eigen::Matrix2d R = T.rotation();
    Eigen::Vector2d pq = T * up - uq;
    Eigen::Matrix2d cinv = (R * cp * R.transpose() + cq).inverse();
    return sqrt(pq.dot(cinv * pq));
  }

  static double Cost2(const Eigen::Vector2d &up,
                      const Eigen::Matrix2d &cp,
                      const Eigen::Vector2d &uq,
                      const Eigen::Matrix2d &cq,
                      const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = Cost(up, cp, uq, cq, T);
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d up_;
  const Eigen::Matrix2d cp_;
  const Eigen::Vector2d uq_;
  const Eigen::Matrix2d cq_;
};

class D2DNDTCostFunctor {
 public:
  D2DNDTCostFunctor(const std::vector<Eigen::Vector2d> &ups,
                    const std::vector<Eigen::Matrix2d> &cps,
                    const std::vector<Eigen::Vector2d> &uqs,
                    const std::vector<Eigen::Matrix2d> &cqs)
      : ups_(ups), cps_(cps), uqs_(uqs), cqs_(cqs), n_(ups.size()) {}
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
      e[0] += -ceres::exp(-0.5 * (m.transpose() * c.inverse() * m)[0]);
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs) {
    return new ceres::AutoDiffFirstOrderFunction<D2DNDTCostFunctor, 3>(
        new D2DNDTCostFunctor(ups, cps, uqs, cqs));
  }

  static double Cost(const std::vector<Eigen::Vector2d> &ups,
                     const std::vector<Eigen::Matrix2d> &cps,
                     const std::vector<Eigen::Vector2d> &uqs,
                     const std::vector<Eigen::Matrix2d> &cqs,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = 0;
    Eigen::Matrix2d R = T.rotation();
    for (size_t i = 0; i < ups.size(); ++i) {
      Eigen::Vector2d pq = T * ups[i] - uqs[i];
      Eigen::Matrix2d cinv = (R * cps[i] * R.transpose() + cqs[i]).inverse();
      cost += -ceres::exp(-0.5 * (pq.dot(cinv * pq)));
    }
    return cost;
  }

 private:
  const std::vector<Eigen::Vector2d> ups_;
  const std::vector<Eigen::Matrix2d> cps_;
  const std::vector<Eigen::Vector2d> uqs_;
  const std::vector<Eigen::Matrix2d> cqs_;
  const int n_;
};

class SNDTMDCostFunctor {
 public:
  SNDTMDCostFunctor(const Eigen::Vector2d &up,
                    const Eigen::Matrix2d &cp,
                    const Eigen::Vector2d &unp,
                    const Eigen::Matrix2d &cnp,
                    const Eigen::Vector2d &uq,
                    const Eigen::Matrix2d &cq,
                    const Eigen::Vector2d &unq,
                    const Eigen::Matrix2d &cnq)
      : up_(up),
        unp_(unp),
        uq_(uq),
        unq_(unq),
        cp_(cp),
        cnp_(cnp),
        cq_(cq),
        cnq_(cnq) {}

  template <typename T>
  bool operator()(const T *const x,
                  const T *const y,
                  const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up = R * up_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> unp = R * unp_.cast<T>();
    Eigen::Matrix<T, 2, 2> cp = R * cp_.cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 2> cnp = R * cnp_.cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 2, 1> unq = unq_.cast<T>();
    Eigen::Matrix<T, 2, 2> cq = cq_.cast<T>();
    Eigen::Matrix<T, 2, 2> cnq = cnq_.cast<T>();
    // FIXME: normalize normal
    // unp2 = unp2.dot(unq2) / ceres::abs(unp2.dot(unq2)) * unp2;

    Eigen::Matrix<T, 2, 1> m1 = up - uq;
    Eigen::Matrix<T, 2, 1> m2 = unp + unq;
    Eigen::Matrix<T, 2, 2> c1 = cp + cq;
    Eigen::Matrix<T, 2, 2> c2 = cnp + cnq;

    T num = m1.dot(m2);
    T den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
    *e = num / den;
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &up,
                                     const Eigen::Matrix2d &cp,
                                     const Eigen::Vector2d &unp,
                                     const Eigen::Matrix2d &cnp,
                                     const Eigen::Vector2d &uq,
                                     const Eigen::Matrix2d &cq,
                                     const Eigen::Vector2d &unq,
                                     const Eigen::Matrix2d &cnq) {
    return new ceres::AutoDiffCostFunction<SNDTMDCostFunctor, 1, 1, 1, 1>(
        new SNDTMDCostFunctor(up, cp, unp, cnp, uq, cq, unq, cnq));
  }

  static double Cost(const Eigen::Vector2d &up,
                     const Eigen::Matrix2d &cp,
                     const Eigen::Vector2d &unp,
                     const Eigen::Matrix2d &cnp,
                     const Eigen::Vector2d &uq,
                     const Eigen::Matrix2d &cq,
                     const Eigen::Vector2d &unq,
                     const Eigen::Matrix2d &cnq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    Eigen::Matrix2d R = T.rotation();
    Eigen::Vector2d m1 = T * up - uq;
    // Eigen::Vector2d unp2 = ((R * unp).dot(unq) > 0) ? unp : -unp;
    Eigen::Vector2d m2 = R * unp + unq;
    Eigen::Matrix2d c1 = R * cp * R.transpose() + cq;
    Eigen::Matrix2d c2 = R * cnp * R.transpose() + cnq;
    return m1.dot(m2) /
           sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
  }

  static double Cost2(const Eigen::Vector2d &up,
                      const Eigen::Matrix2d &cp,
                      const Eigen::Vector2d &unp,
                      const Eigen::Matrix2d &cnp,
                      const Eigen::Vector2d &uq,
                      const Eigen::Matrix2d &cq,
                      const Eigen::Vector2d &unq,
                      const Eigen::Matrix2d &cnq,
                      const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = Cost(up, cp, unp, cnp, uq, cq, unq, cnq, T);
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d up_, unp_, uq_, unq_;
  const Eigen::Matrix2d cp_, cnp_, cq_, cnq_;
};

class SNDTMDCostFunctor2 {
 public:
  SNDTMDCostFunctor2(const Eigen::Vector2d &up,
                     const Eigen::Matrix2d &cp,
                     const Eigen::Vector2d &unp,
                     const Eigen::Vector2d &uq,
                     const Eigen::Matrix2d &cq,
                     const Eigen::Vector2d &unq)
      : up_(up), unp_(unp), uq_(uq), unq_(unq), cp_(cp), cq_(cq) {}

  template <typename T>
  bool operator()(const T *const x,
                  const T *const y,
                  const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up = R * up_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> unp = R * unp_.cast<T>();
    Eigen::Matrix<T, 2, 2> cp = R * cp_.cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 2, 1> unq = unq_.cast<T>();
    Eigen::Matrix<T, 2, 2> cq = cq_.cast<T>();
    unp = unp.dot(unq) / ceres::abs(unp.dot(unq)) * unp;

    Eigen::Matrix<T, 2, 1> m1 = up - uq;
    Eigen::Matrix<T, 2, 1> m2 = unp + unq;
    Eigen::Matrix<T, 2, 2> c1 = cp + cq;

    T num = m1.dot(m2);
    T den = sqrt(m2.dot(c1 * m2));
    *e = num / den;
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &up,
                                     const Eigen::Matrix2d &cp,
                                     const Eigen::Vector2d &unp,
                                     const Eigen::Vector2d &uq,
                                     const Eigen::Matrix2d &cq,
                                     const Eigen::Vector2d &unq) {
    return new ceres::AutoDiffCostFunction<SNDTMDCostFunctor2, 1, 1, 1, 1>(
        new SNDTMDCostFunctor2(up, cp, unp, uq, cq, unq));
  }

  static double Cost(const Eigen::Vector2d &up,
                     const Eigen::Matrix2d &cp,
                     const Eigen::Vector2d &unp,
                     const Eigen::Vector2d &uq,
                     const Eigen::Matrix2d &cq,
                     const Eigen::Vector2d &unq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    Eigen::Matrix2d R = T.rotation();
    Eigen::Vector2d m1 = T * up - uq;
    Eigen::Vector2d m2 = R * unp + unq;
    Eigen::Matrix2d c1 = R * cp * R.transpose() + cq;
    return m1.dot(m2) / sqrt(m2.dot(c1 * m2));
  }

  static double Cost2(const Eigen::Vector2d &up,
                      const Eigen::Matrix2d &cp,
                      const Eigen::Vector2d &unp,
                      const Eigen::Vector2d &uq,
                      const Eigen::Matrix2d &cq,
                      const Eigen::Vector2d &unq,
                      const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = Cost(up, cp, unp, uq, cq, unq, T);
    return 0.5 * cost * cost;
  }

 private:
  const Eigen::Vector2d up_, unp_, uq_, unq_;
  const Eigen::Matrix2d cp_, cq_;
};

class SNDTCostFunctor2 {
 public:
  SNDTCostFunctor2(const std::vector<Eigen::Vector2d> &ups,
                   const std::vector<Eigen::Matrix2d> &cps,
                   const std::vector<Eigen::Vector2d> &unps,
                   const std::vector<Eigen::Vector2d> &uqs,
                   const std::vector<Eigen::Matrix2d> &cqs,
                   const std::vector<Eigen::Vector2d> &unqs)
      : ups_(ups),
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
      unp = unp.dot(unq) / ceres::abs(unp.dot(unq)) * unp;

      Eigen::Matrix<T, 2, 1> m1 = up - uq;
      Eigen::Matrix<T, 2, 1> m2 = unp + unq;
      Eigen::Matrix<T, 2, 2> c1 = cp + cq;

      T num = m1.dot(m2);
      T den = sqrt(m2.dot(c1 * m2));
      *e += -ceres::exp(-0.5 * ceres::pow(num / den, 2));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &unps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs,
      const std::vector<Eigen::Vector2d> &unqs) {
    return new ceres::AutoDiffFirstOrderFunction<SNDTCostFunctor2, 3>(
        new SNDTCostFunctor2(ups, cps, unps, uqs, cqs, unqs));
  }

  static double Cost(const std::vector<Eigen::Vector2d> &ups,
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
      Eigen::Vector2d m2 = R * unps[i] + unqs[i];
      Eigen::Matrix2d c1 = R * cps[i] * R.transpose() + cqs[i];
      cost += -ceres::exp(-0.5 * m1.dot(m2) * m1.dot(m2) / m2.dot(c1 * m2));
    }
    return cost;
  }

 private:
  const std::vector<Eigen::Vector2d> ups_, unps_, uqs_, unqs_;
  const std::vector<Eigen::Matrix2d> cps_, cqs_;
  const int n_;
};
