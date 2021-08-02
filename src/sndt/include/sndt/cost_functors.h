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

    *e = abs((p2 - q2).dot(np2 + nq2));
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

  template <typename T>
  static Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw) {
    T cos_yaw = ceres::cos(yaw);
    T sin_yaw = ceres::sin(yaw);
    Eigen::Matrix<T, 2, 2> ret;
    ret << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
    return ret;
  }

  static ceres::CostFunction *Create(const SNDTCell *cellp,
                                     const SNDTCell *cellq) {
    return new ceres::AutoDiffCostFunction<SNDTCostFunctor, 1, 1, 1, 1>(
        new SNDTCostFunctor(cellp, cellq));
  }

  static double Score(const SNDTCell *p, const SNDTCell *q) {
    Eigen::Vector2d up = p->GetPointMean(), uq = q->GetPointMean();
    Eigen::Vector2d unp = p->GetNormalMean(), unq = q->GetNormalMean();
    Eigen::Matrix2d cp = p->GetPointCov(), cq = q->GetPointCov();
    Eigen::Matrix2d cnp = p->GetNormalCov(), cnq = q->GetNormalCov();
    Eigen::Vector2d m1 = up - uq, m2 = unp + unq;
    Eigen::Matrix2d c1 = cp + cq, c2 = cnp + cnq;
    double num = m1.dot(m2);
    double den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
    return num / den;
  }

  static std::vector<double> MeanAndCov(const SNDTCell *p, const SNDTCell *q) {
    Eigen::Vector2d up = p->GetPointMean(), uq = q->GetPointMean();
    Eigen::Vector2d unp = p->GetNormalMean(), unq = q->GetNormalMean();
    Eigen::Matrix2d cp = p->GetPointCov(), cq = q->GetPointCov();
    Eigen::Matrix2d cnp = p->GetNormalCov(), cnq = q->GetNormalCov();
    Eigen::Vector2d m1 = up - uq, m2 = unp + unq;
    Eigen::Matrix2d c1 = cp + cq, c2 = cnp + cnq;
    return {m1.dot(m2), m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace()};
  }
};
