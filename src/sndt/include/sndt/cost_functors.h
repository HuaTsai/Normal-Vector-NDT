#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sndt/ndt_cell.h>
#include <sndt/sndt_cell.h>

template <typename T>
Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw) {
  T cos_yaw = ceres::cos(yaw);
  T sin_yaw = ceres::sin(yaw);
  Eigen::Matrix<T, 2, 2> ret;
  ret << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return ret;
}

class AngleLocalParameterization {
 public:
  template <typename T>
  bool operator()(const T *t, const T *dt, T *t_add_dt) const {
    T pi(M_PI), twopi(2. * M_PI), sum(*t + *dt);
    *t_add_dt = sum - twopi * ceres::floor((sum + pi) / twopi);
    return true;
  }

  static ceres::LocalParameterization *Create() {
    return new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>;
  }
};

struct ICPCostFunctor {
  const Eigen::Vector2d p, q;
  ICPCostFunctor(const Eigen::Vector2d &p_, const Eigen::Vector2d &q_)
      : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p2 = R * p.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> q2 = q.cast<T>();

    *e = (p2 - q2).norm();
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &q) {
    return new ceres::AutoDiffCostFunction<ICPCostFunctor, 1, 1, 1, 1>(
        new ICPCostFunctor(p, q));
  }
};

struct Pt2plICPCostFunctor {
  const Eigen::Vector2d p, q, nq;
  Pt2plICPCostFunctor(const Eigen::Vector2d &p_, const Eigen::Vector2d &q_,
                      const Eigen::Vector2d &nq_)
      : p(p_), q(q_), nq(nq_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p2 = R * p.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> q2 = q.cast<T>();

    Eigen::Matrix<T, 2, 1> nq2 = nq.cast<T>();

    *e = (p2 - q2).dot(nq2);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &q,
                                     const Eigen::Vector2d &nq) {
    return new ceres::AutoDiffCostFunction<Pt2plICPCostFunctor, 1, 1, 1, 1>(
        new Pt2plICPCostFunctor(p, q, nq));
  }
};

struct SICPCostFunctor {
  const Eigen::Vector2d p, np, q, nq;
  SICPCostFunctor(const Eigen::Vector2d &p_, const Eigen::Vector2d &np_,
                  const Eigen::Vector2d &q_, const Eigen::Vector2d &nq_)
      : p(p_), np(np_), q(q_), nq(nq_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p2 = R * p.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> np2 = R * np.cast<T>();

    Eigen::Matrix<T, 2, 1> q2 = q.cast<T>();
    Eigen::Matrix<T, 2, 1> nq2 = nq.cast<T>();

    *e = (p2 - q2).dot(np2 + nq2);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &np,
                                     const Eigen::Vector2d &q,
                                     const Eigen::Vector2d &nq) {
    return new ceres::AutoDiffCostFunction<SICPCostFunctor, 1, 1, 1, 1>(
        new SICPCostFunctor(p, np, q, nq));
  }
};

struct NDTP2DCostFunctor {
  const Eigen::Vector2d p;
  const NDTCell *q;
  NDTP2DCostFunctor(const Eigen::Vector2d &p_, const NDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p2 = R * p.cast<T>() + t;

    Eigen::Matrix<T, 2, 1> uq = q->GetPointMean().cast<T>();
    Eigen::Matrix<T, 2, 2> cq = q->GetPointCov().cast<T>();

    *e = sqrt(((p2 - uq).transpose() * cq.inverse() * (p2 - uq))[0]);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const NDTCell *cellq) {
    return new ceres::AutoDiffCostFunction<NDTP2DCostFunctor, 1, 1, 1, 1>(
        new NDTP2DCostFunctor(p, cellq));
  }
};

struct NDTD2DCostFunctor {
  const NDTCell *p, *q;
  NDTD2DCostFunctor(const NDTCell *p_, const NDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up = R * p->GetPointMean().cast<T>() + t;
    Eigen::Matrix<T, 2, 2> cp = R * p->GetPointCov().cast<T>() * R.transpose();

    Eigen::Matrix<T, 2, 1> uq = q->GetPointMean().cast<T>();
    Eigen::Matrix<T, 2, 2> cq = q->GetPointCov().cast<T>();

    Eigen::Matrix<T, 2, 1> m = up - uq;
    Eigen::Matrix<T, 2, 2> c = cp + cq;

    *e = sqrt((m.transpose() * c.inverse() * m)[0]);
    return true;
  }

  static ceres::CostFunction *Create(const NDTCell *cellp,
                                     const NDTCell *cellq) {
    return new ceres::AutoDiffCostFunction<NDTD2DCostFunctor, 1, 1, 1, 1>(
        new NDTD2DCostFunctor(cellp, cellq));
  }
};

struct SNDTCostFunctor {
  const SNDTCell *p, *q;
  SNDTCostFunctor(const SNDTCell *p_, const SNDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up = R * p->GetPointMean().cast<T>() + t;
    Eigen::Matrix<T, 2, 1> unp = R * p->GetNormalMean().cast<T>();
    Eigen::Matrix<T, 2, 2> cp = R * p->GetPointCov().cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 2> cnp = R * p->GetNormalCov().cast<T>() * R.transpose();

    Eigen::Matrix<T, 2, 1> uq = q->GetPointMean().cast<T>();
    Eigen::Matrix<T, 2, 1> unq = q->GetNormalMean().cast<T>();
    Eigen::Matrix<T, 2, 2> cq = q->GetPointCov().cast<T>();
    Eigen::Matrix<T, 2, 2> cnq = q->GetNormalCov().cast<T>();

    Eigen::Matrix<T, 2, 1> m1 = up - uq;
    Eigen::Matrix<T, 2, 1> m2 = unp + unq;
    Eigen::Matrix<T, 2, 2> c1 = cp + cq;
    Eigen::Matrix<T, 2, 2> c2 = cnp + cnq;

    T num = m1.dot(m2);
    T den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
    *e = num / den;
    return true;
  }

  static ceres::CostFunction *Create(const SNDTCell *cellp,
                                     const SNDTCell *cellq) {
    return new ceres::AutoDiffCostFunction<SNDTCostFunctor, 1, 1, 1, 1>(
        new SNDTCostFunctor(cellp, cellq));
  }
};

struct SNDTCostFunctor2 {
  const Eigen::Vector2d up, unp, uq, unq;
  const Eigen::Matrix2d cp, cnp, cq, cnq;
  SNDTCostFunctor2(const Eigen::Vector2d &up_, const Eigen::Matrix2d &cp_,
                   const Eigen::Vector2d &unp_, const Eigen::Matrix2d &cnp_,
                   const Eigen::Vector2d &uq_, const Eigen::Matrix2d &cq_,
                   const Eigen::Vector2d &unq_, const Eigen::Matrix2d &cnq_)
      : up(up_), unp(unp_), uq(uq_), unq(unq_), cp(cp_), cnp(cnp_), cq(cq_), cnq(cnq_) {}

  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up2 = R * up.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> unp2 = R * unp.cast<T>();
    Eigen::Matrix<T, 2, 2> cp2 = R * cp.cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 2> cnp2 = R * cnp.cast<T>() * R.transpose();

    Eigen::Matrix<T, 2, 1> uq2 = uq.cast<T>();
    Eigen::Matrix<T, 2, 1> unq2 = unq.cast<T>();
    Eigen::Matrix<T, 2, 2> cq2 = cq.cast<T>();
    Eigen::Matrix<T, 2, 2> cnq2 = cnq.cast<T>();

    Eigen::Matrix<T, 2, 1> m1 = up2 - uq2;
    Eigen::Matrix<T, 2, 1> m2 = unp2 + unq2;
    Eigen::Matrix<T, 2, 2> c1 = cp2 + cq2;
    Eigen::Matrix<T, 2, 2> c2 = cnp2 + cnq2;

    T num = m1.dot(m2);
    T den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
    *e = num / den;
    return true;
  }

  static ceres::CostFunction *Create(
      const Eigen::Vector2d &up, const Eigen::Matrix2d &cp,
      const Eigen::Vector2d &unp, const Eigen::Matrix2d &cnp,
      const Eigen::Vector2d &uq, const Eigen::Matrix2d &cq,
      const Eigen::Vector2d &unq, const Eigen::Matrix2d &cnq) {
    return new ceres::AutoDiffCostFunction<SNDTCostFunctor2, 1, 1, 1, 1>(
        new SNDTCostFunctor2(up, cp, unp, cnp, uq, cq, unq, cnq));
  }
};

struct SNDTCostFunctor3 {
  const Eigen::Vector2d up, unp, uq, unq;
  const Eigen::Matrix2d cp, cq;
  SNDTCostFunctor3(const Eigen::Vector2d &up_, const Eigen::Matrix2d &cp_,
                   const Eigen::Vector2d &unp_, const Eigen::Vector2d &uq_,
                   const Eigen::Matrix2d &cq_, const Eigen::Vector2d &unq_)
      : up(up_), unp(unp_), uq(uq_), unq(unq_), cp(cp_), cq(cq_) {}

  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up2 = R * up.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> unp2 = R * unp.cast<T>();
    Eigen::Matrix<T, 2, 2> cp2 = R * cp.cast<T>() * R.transpose();

    Eigen::Matrix<T, 2, 1> uq2 = uq.cast<T>();
    Eigen::Matrix<T, 2, 1> unq2 = unq.cast<T>();
    Eigen::Matrix<T, 2, 2> cq2 = cq.cast<T>();

    Eigen::Matrix<T, 2, 1> m1 = up2 - uq2;
    Eigen::Matrix<T, 2, 1> m2 = unp2 + unq2;
    Eigen::Matrix<T, 2, 2> c1 = cp2 + cq2;

    T num = m1.dot(m2);
    T den = sqrt(m2.dot(c1 * m2));
    *e = num / den;
    return true;
  }

  static ceres::CostFunction *Create(
      const Eigen::Vector2d &up, const Eigen::Matrix2d &cp,
      const Eigen::Vector2d &unp, const Eigen::Vector2d &uq,
      const Eigen::Matrix2d &cq, const Eigen::Vector2d &unq) {
    return new ceres::AutoDiffCostFunction<SNDTCostFunctor3, 1, 1, 1, 1>(
        new SNDTCostFunctor3(up, cp, unp, uq, cq, unq));
  }
};
