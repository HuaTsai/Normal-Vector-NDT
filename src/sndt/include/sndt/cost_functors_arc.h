class P2DNDTMDCostFunctor {
 public:
  P2DNDTMDCostFunctor(const Eigen::Vector2d &p,
                      const Eigen::Vector2d &uq,
                      const Eigen::Matrix2d &cq)
      : p_(p), uq_(uq), cq_(cq) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 2, 2> cq = cq_.cast<T>();
    Eigen::Matrix<T, 2, 1> pq = p - uq;

    *e = sqrt(pq.dot(cq.inverse() * pq));
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &uq,
                                     const Eigen::Matrix2d &cq) {
    return new ceres::AutoDiffCostFunction<P2DNDTMDCostFunctor, 1, 3>(
        new P2DNDTMDCostFunctor(p, uq, cq));
  }

  static double Cost(const Eigen::Vector2d &p,
                     const Eigen::Vector2d &uq,
                     const Eigen::Matrix2d &cq,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    Eigen::Vector2d pq = T * p - uq;
    Eigen::Matrix2d cinv = cq.inverse();
    return 0.5 * pq.dot(cinv * pq);
  }

 private:
  const Eigen::Vector2d p_, uq_;
  const Eigen::Matrix2d cq_;
};

class P2DNDTCostFunctor {
 public:
  P2DNDTCostFunctor(double d2,
                    const std::vector<Eigen::Vector2d> &ps,
                    const std::vector<Eigen::Vector2d> &qs,
                    const std::vector<Eigen::Matrix2d> &cqs)
      : d2_(d2), ps_(ps), qs_(qs), cqs_(cqs), n_(ps.size()) {}
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
      e[0] += -ceres::exp(-d2_ * pq.dot(cq.inverse() * pq));
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      double d2,
      const std::vector<Eigen::Vector2d> &ps,
      const std::vector<Eigen::Vector2d> &qs,
      const std::vector<Eigen::Matrix2d> &cqs) {
    return new ceres::AutoDiffFirstOrderFunction<P2DNDTCostFunctor, 3>(
        new P2DNDTCostFunctor(d2, ps, qs, cqs));
  }

  static double Cost(double d2,
                     const std::vector<Eigen::Vector2d> &ps,
                     const std::vector<Eigen::Vector2d> &qs,
                     const std::vector<Eigen::Matrix2d> &cqs,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
      Eigen::Vector2d pq = T * ps[i] - qs[i];
      Eigen::Matrix2d cinv = cqs[i].inverse();
      cost += -ceres::exp(-d2 * (pq.dot(cinv * pq)));
    }
    return cost;
  }

 private:
  const double d2_;
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
  bool operator()(const T *const xyt,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

    Eigen::Matrix<T, 2, 1> up = R * up_.cast<T>() + t;
    Eigen::Matrix<T, 2, 2> cp = R * cp_.cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 1> uq = uq_.cast<T>();
    Eigen::Matrix<T, 2, 2> cq = cq_.cast<T>();
    Eigen::Matrix<T, 2, 1> m = up - uq;
    Eigen::Matrix<T, 2, 2> c = cp + cq;

    *e = sqrt(m.dot(c.inverse() * m));
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &up,
                                     const Eigen::Matrix2d &cp,
                                     const Eigen::Vector2d &uq,
                                     const Eigen::Matrix2d &cq) {
    return new ceres::AutoDiffCostFunction<D2DNDTMDCostFunctor, 1, 3>(
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
    return 0.5 * pq.dot(cinv * pq);
  }

 private:
  const Eigen::Vector2d up_;
  const Eigen::Matrix2d cp_;
  const Eigen::Vector2d uq_;
  const Eigen::Matrix2d cq_;
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
  bool operator()(const T *const xyt, T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

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
    return new ceres::AutoDiffCostFunction<SNDTMDCostFunctor, 1, 3>(
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
    double num = m1.dot(m2) * m1.dot(m2);
    double den = m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace();
    return 0.5 * num / den;
  }

 private:
  const Eigen::Vector2d up_, unp_, uq_, unq_;
  const Eigen::Matrix2d cp_, cnp_, cq_, cnq_;
};

class SNDTCostFunctor {
 public:
  SNDTCostFunctor(double d2,
                  const std::vector<Eigen::Vector2d> &ups,
                  const std::vector<Eigen::Matrix2d> &cps,
                  const std::vector<Eigen::Vector2d> &unps,
                  const std::vector<Eigen::Matrix2d> &cnps,
                  const std::vector<Eigen::Vector2d> &uqs,
                  const std::vector<Eigen::Matrix2d> &cqs,
                  const std::vector<Eigen::Vector2d> &unqs,
                  const std::vector<Eigen::Matrix2d> &cnqs)
      : d2_(d2),
        ups_(ups),
        unps_(unps),
        uqs_(uqs),
        unqs_(unqs),
        cps_(cps),
        cnps_(cnps),
        cqs_(cqs),
        cnqs_(cnqs),
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
      Eigen::Matrix<T, 2, 2> cnp = R * cnps_[i].cast<T>() * R.transpose();
      Eigen::Matrix<T, 2, 1> uq = uqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 1> unq = unqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cq = cqs_[i].cast<T>();
      Eigen::Matrix<T, 2, 2> cnq = cnqs_[i].cast<T>();
      // FIXME: normalize normal
      // unp2 = unp2.dot(unq2) / ceres::abs(unp2.dot(unq2)) * unp2;

      Eigen::Matrix<T, 2, 1> m1 = up - uq;
      Eigen::Matrix<T, 2, 1> m2 = unp + unq;
      Eigen::Matrix<T, 2, 2> c1 = cp + cq;
      Eigen::Matrix<T, 2, 2> c2 = cnp + cnq;

      T num2 = m1.dot(m2) * m1.dot(m2);
      T den2 = m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace();
      e[0] += -ceres::exp(-d2_ * num2 / den2);
    }
    return true;
  }

  static ceres::FirstOrderFunction *Create(
      double d2,
      const std::vector<Eigen::Vector2d> &ups,
      const std::vector<Eigen::Matrix2d> &cps,
      const std::vector<Eigen::Vector2d> &unps,
      const std::vector<Eigen::Matrix2d> &cnps,
      const std::vector<Eigen::Vector2d> &uqs,
      const std::vector<Eigen::Matrix2d> &cqs,
      const std::vector<Eigen::Vector2d> &unqs,
      const std::vector<Eigen::Matrix2d> &cnqs) {
    return new ceres::AutoDiffFirstOrderFunction<SNDTCostFunctor, 3>(
        new SNDTCostFunctor(d2, ups, cps, unps, cnps, uqs, cqs, unqs, cnqs));
  }

  static double Cost(double d2,
                     const std::vector<Eigen::Vector2d> &ups,
                     const std::vector<Eigen::Matrix2d> &cps,
                     const std::vector<Eigen::Vector2d> &unps,
                     const std::vector<Eigen::Matrix2d> &cnps,
                     const std::vector<Eigen::Vector2d> &uqs,
                     const std::vector<Eigen::Matrix2d> &cqs,
                     const std::vector<Eigen::Vector2d> &unqs,
                     const std::vector<Eigen::Matrix2d> &cnqs,
                     const Eigen::Affine2d &T = Eigen::Affine2d::Identity()) {
    double cost = 0;
    Eigen::Matrix2d R = T.rotation();
    for (size_t i = 0; i < ups.size(); ++i) {
      Eigen::Vector2d m1 = T * ups[i] - uqs[i];
      // Eigen::Vector2d unp2 = ((R * unp).dot(unq) > 0) ? unp : -unp;
      Eigen::Vector2d m2 = R * unps[i] + unqs[i];
      Eigen::Matrix2d c1 = R * cps[i] * R.transpose() + cqs[i];
      Eigen::Matrix2d c2 = R * cnps[i] * R.transpose() + cnqs[i];
      double num2 = m1.dot(m2) * m1.dot(m2);
      double den2 = m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace();
      cost += -ceres::exp(-d2 * num2 / den2);
    }
    return cost;
  }

 private:
  const double d2_;
  const std::vector<Eigen::Vector2d> ups_, unps_, uqs_, unqs_;
  const std::vector<Eigen::Matrix2d> cps_, cnps_, cqs_, cnqs_;
  const int n_;
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
  bool operator()(const T *const xyt, T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);

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
    return new ceres::AutoDiffCostFunction<SNDTMDCostFunctor2, 1, 3>(
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
    double num = m1.dot(m2) * m1.dot(m2);
    double den = m2.dot(c1 * m2);
    return 0.5 * num / den;
  }

 private:
  const Eigen::Vector2d up_, unp_, uq_, unq_;
  const Eigen::Matrix2d cp_, cq_;
};
