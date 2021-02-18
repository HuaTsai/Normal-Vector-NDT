#pragma once

#include <bits/stdc++.h>
#include <pcl_ros/point_cloud.h>
#include <sndt/ndt_map_2d.hpp>
#include <ceres/ceres.h>

using namespace std;

// #define USE_CLOSEST_NEIGHBOR

struct CostFunction {
  NDTCell *p, *q;
  CostFunction(NDTCell *p_, NDTCell *q_) : p(p_), q(q_) {}
  template<typename T>
  bool operator()(const T *const xyt, T *e) const {
    T th = xyt[2];
    T m1[2] = {cos(th) * p->getPointMean()(0) - sin(th) * p->getPointMean()(1) + xyt[0] - q->getPointMean()(0),
               sin(th) * p->getPointMean()(0) + cos(th) * p->getPointMean()(1) + xyt[1] - q->getPointMean()(1)};
    T m2[2] = {cos(th) * p->getNormalMean()(0) - sin(th) * p->getNormalMean()(1) + q->getNormalMean()(0),
               sin(th) * p->getNormalMean()(0) + cos(th) * p->getNormalMean()(1) + q->getNormalMean()(1)};
    double ap = p->getPointCov()(0, 0);
    double bp = p->getPointCov()(0, 1);
    double cp = p->getPointCov()(1, 1);
    double anp = p->getNormalCov()(0, 0);
    double bnp = p->getNormalCov()(0, 1);
    double cnp = p->getNormalCov()(1, 1);
    double aq = q->getPointCov()(0, 0);
    double bq = q->getPointCov()(0, 1);
    double cq = q->getPointCov()(1, 1);
    double anq = q->getNormalCov()(0, 0);
    double bnq = q->getNormalCov()(0, 1);
    double cnq = q->getNormalCov()(1, 1);
    T c1[3] = {ap * cos(th) * cos(th) - 2. * bp * sin(th) * cos(th) + cp * sin(th) * sin(th) + aq,
               ap * sin(th) * cos(th) - bp * sin(th) * sin(th) + bp * cos(th) * cos(th) - cp * sin(th) * cos(th) + bq,
               ap * sin(th) * sin(th) + 2. * bp * sin(th) * cos(th) + cp * cos(th) * cos(th) + cq};
    T c2[3] = {anp * cos(th) * cos(th) - 2. * bnp * sin(th) * cos(th) + cnp * sin(th) * sin(th) + anq,
               anp * sin(th) * cos(th) - bnp * sin(th) * sin(th) + bnp * cos(th) * cos(th) - cnp * sin(th) * cos(th) + bnq,
               anp * sin(th) * sin(th) + 2. * bnp * sin(th) * cos(th) + cnp * cos(th) * cos(th) + cnq};
    T f = (m1[0] * m2[0] + m1[1] * m2[1]) * (m1[0] * m2[0] + m1[1] * m2[1]);
    T g = (c2[0] * m1[0] * m1[0] + 2. * c2[1] * m1[0] * m1[1] + c2[2] * m1[1] * m1[1]) +
          (c1[0] * m2[0] * m2[0] + 2. * c1[1] * m2[0] * m2[1] + c1[2] * m2[1] * m2[1]) +
          (c1[0] * c2[0] + 2. * c1[1] * c2[1] + c1[2] * c2[2]);
    T l = f / g;
    // e[0] = -exp(l / 2.);
    e[0] = -l;
    return true;
  }
};

class NDTMatcherD2D_2D {
 public:
  NDTMatcherD2D_2D() {
    d1 = 1;
    d2 = 0.05;
    itr_max = 100;
    delta_score = 1e-1;
    step_control = false;
    step_size = 1;
    neighbors = 2;
  }

  bool match(NDTMap &targetNDT, NDTMap &sourceNDT,
             Eigen::Matrix3d &T, bool useInitialGuess = true) {
    bool convergence = false;
    int itr_ctr = 0;
    Eigen::Matrix3d H;
    Eigen::Vector3d g, deltax;
    Eigen::Matrix3d TR, Tbest;
    Eigen::Vector3d transformed_vec, mean;

    bool ret = true;
    double score_best = INT_MAX;
    if (!useInitialGuess) {
      T.setIdentity();
    }
    Tbest = T;

    vector<NDTCell *> nextNDT = sourceNDT.pseudoTransformNDT(T);
    while (!convergence) {
      dprintf("iter: %d\n", itr_ctr);
      TR.setIdentity();
      double score_here = derivativesNDT(nextNDT, targetNDT, g, H);
      cerr << "grad: " << g.transpose() << endl;
      cerr << "hess: " << endl << H << endl;
      cerr << "score_here: " << score_here << endl;
      if (score_here < score_best) {
        Tbest = T;
        score_best = score_here;
      }

      if (g.norm() <= 10e-2 * delta_score) {
        dprintf("g.norm() = %f too small\n", g.norm());
        for (unsigned int i = 0; i < nextNDT.size(); i++) {
          if (nextNDT[i] != NULL) delete nextNDT[i];
        }
        if (score_here > score_best) {
          cerr << "crap iterations, best was " << score_best << " last was "
               << score_here << endl;
          T = Tbest;
        }
        return true;
      }

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> Sol(H);
      Eigen::Vector3d evals = Sol.eigenvalues().real();
      double minCoeff = evals.minCoeff();
      double maxCoeff = evals.maxCoeff();

      dprintf("minCoeff: %f, maxCoeff: %f\n", minCoeff, maxCoeff);
      if (minCoeff < 10e-1) {
        dprintf("minCoeff too small\n");
        Eigen::Matrix3d evecs = Sol.eigenvectors().real();
        double regularizer = g.norm();
        regularizer = regularizer + minCoeff > 0 ? regularizer : 0.001 * maxCoeff - minCoeff;
        Eigen::Vector3d reg;
        reg << regularizer, regularizer, regularizer;
        evals += reg;
        Eigen::Matrix3d Lam;
        Lam = evals.asDiagonal();
        H = evecs * Lam * (evecs.transpose());
      }

      deltax = -H.ldlt().solve(g);
      double dginit = deltax.dot(g);
      cerr << "deltax: " << deltax.transpose() << endl;

      if (dginit > 0) {
        dprintf("dginit > 0\n");
        for (unsigned int i = 0; i < nextNDT.size(); i++) {
          if (nextNDT[i] != NULL) delete nextNDT[i];
        }
        if (score_here > score_best) {
          T = Tbest;
        }
        return true;
      }

      if (step_control) {
        // step_size = lineSearch2D(deltax, nextNDT, targetNDT);
      } else {
        dprintf("Use step_size = 1\n");
        step_size = 1;
      }
      deltax = step_size * deltax;

      TR << cos(deltax(2)), -sin(deltax(2)), deltax(0),
            sin(deltax(2)), cos(deltax(2)), deltax(1),
            0, 0, 1;
      T = TR * T;
      cerr << "T = " << endl << T << endl;

      for (unsigned int i = 0; i < nextNDT.size(); i++) {
        Eigen::Matrix2d R = TR.block<2, 2>(0, 0);
        Eigen::Vector2d t = TR.block<2, 1>(0, 2);
        Eigen::Vector2d mean = nextNDT[i]->getPointMean();
        Eigen::Matrix2d cov = nextNDT[i]->getPointCov();
        mean = R * mean + t;
        cov = R * cov * R.transpose();
        nextNDT[i]->setPointMean(mean);
        nextNDT[i]->setPointCov(cov);

        mean = nextNDT[i]->getNormalMean();
        cov = nextNDT[i]->getNormalCov();
        mean = R * mean;
        cov = R * cov * R.transpose();
        nextNDT[i]->setNormalMean(mean);
        nextNDT[i]->setNormalCov(cov);
      }

      if (itr_ctr > 0) {
        convergence = (deltax.norm() < delta_score);
      }

      if (itr_ctr > itr_max) {
        convergence = true;
        ret = false;
      }

      ++itr_ctr;
    }

    double score_here = derivativesNDT(nextNDT, targetNDT, g, H);
    if (score_here > score_best) {
      T = Tbest;
    }
    cerr << "score best: " << score_best << endl;
    for (unsigned int i = 0; i < nextNDT.size(); i++) {
      if (nextNDT[i] != NULL) delete nextNDT[i];
    }
    return ret;
  }

  bool ceresmatch(NDTMap &targetNDT, NDTMap &sourceNDT,
                  Eigen::Matrix3d &T, bool useInitialGuess = true) {
    if (!useInitialGuess) {
      T.setIdentity();
    }

    bool converge = false;
    int itr = 0;
    while (!converge) {
      ++itr;
      double xyt[3];
      vector<NDTCell *> nextNDT = sourceNDT.pseudoTransformNDT(T);
      ceres::Problem problem;
      for (size_t i = 0; i < nextNDT.size(); i++) {
        NDTCell *cellp = nextNDT.at(i);
        pcl::PointXY pt;
        pt.x = cellp->getPointMean()(0);
        pt.y = cellp->getPointMean()(1);
        vector<NDTCell *> cells = targetNDT.getCellsForPoint(pt, neighbors, true);
        for (size_t j = 0; j < cells.size(); ++j) {
          NDTCell *cellq = cells.at(j);
          if (cellq == NULL) {
            continue;
          }
          if (cellq->phasGaussian_ && cellq->nhasGaussian_) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
                    new CostFunction(cellp, cellq)),
                nullptr, xyt);
          }
        }
      }
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      Eigen::Matrix3d dT = Eigen::Matrix3d::Identity();
      double th = xyt[2];
      dT.block<2, 2>(0, 0) << cos(th), -sin(th), sin(th), cos(th);
      dT(0, 2) = xyt[0];
      dT(1, 2) = xyt[1];
      Eigen::Vector3d vxyt(xyt[0], xyt[1], xyt[2]);
      converge = (vxyt.norm() < 0.2) || (itr > 100);
      // printf("xyt: %f, %f, %f -> %f\n", xyt[0], xyt[1], xyt[2], vxyt.norm());
      T = dT * T;
      dprintf("  itr(%d): (%.2f, %.2f, %.2f)\n", itr, T(0, 2), T(1, 2), acos(T(0, 0)) * 180. / M_PI);
    }
    return true;
  }

  bool ceresmatchnn(NDTMap &targetNDT, NDTMap &sourceNDT,
                    Eigen::Matrix3d &T, bool useInitialGuess = true) {
    if (!useInitialGuess) {
      T.setIdentity();
    }

    bool converge = false;
    int itr = 0;
    while (!converge) {
      ++itr;
      double xyt[3];
      vector<NDTCell *> nextNDT = sourceNDT.pseudoTransformNDT(T);
      ceres::Problem problem;
      for (size_t i = 0; i < nextNDT.size(); i++) {
        NDTCell *cellp = nextNDT.at(i);
        pcl::PointXY pt;
        pt.x = cellp->getPointMean()(0);
        pt.y = cellp->getPointMean()(1);
        NDTCell *cellq;
        targetNDT.getCellForPoint(pt, cellq, neighbors, true);
        if (cellq == NULL) {
          continue;
        }
        if (cellq->phasGaussian_ && cellq->nhasGaussian_) {
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
                  new CostFunction(cellp, cellq)),
              nullptr, xyt);
        }
      }
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      Eigen::Matrix3d dT = Eigen::Matrix3d::Identity();
      double th = xyt[2];
      dT.block<2, 2>(0, 0) << cos(th), -sin(th), sin(th), cos(th);
      dT(0, 2) = xyt[0];
      dT(1, 2) = xyt[1];
      Eigen::Vector3d vxyt(xyt[0], xyt[1], xyt[2]);
      converge = (vxyt.norm() < 0.2) || (itr > 100);
      // printf("xyt: %f, %f, %f -> %f\n", xyt[0], xyt[1], xyt[2], vxyt.norm());
      T = dT * T;
      dprintf("  itr(%d): (%.2f, %.2f, %.2f)\n", itr, T(0, 2), T(1, 2), acos(T(0, 0)) * 180. / M_PI);
    }
    return true;
  }

  double derivativesNDT(const vector<NDTCell *> &nextNDT,
                        const NDTMap &targetNDT, Eigen::Vector3d &gs,
                        Eigen::Matrix3d &Hs) {
    double scores = 0;
    gs.setZero();
    Hs.setZero();
    dprintf("next size: %ld\n", nextNDT.size());
    for (size_t i = 0; i < nextNDT.size(); i++) {
      Eigen::Vector2d mu_p, mu_np;
      Eigen::Matrix2d sigma_p, sigma_np;
      mu_p = nextNDT.at(i)->getPointMean();
      sigma_p = nextNDT.at(i)->getPointCov();
      mu_np = nextNDT.at(i)->getNormalMean();
      sigma_np = nextNDT.at(i)->getNormalCov();
      basicDerivatives(mu_p, sigma_p, mu_np, sigma_np);

      pcl::PointXY pt;
      pt.x = mu_p(0);
      pt.y = mu_p(1);

#ifdef USE_CLOSEST_NEIGHBOR
      NDTCell *cell = NULL;
      targetNDT.getCellForPoint(pt, cell, neighbors, true);
      if (cell == NULL) {
        continue;
      }
      if (cell->phasGaussian_ && cell->nhasGaussian_) {
        Eigen::Vector2d mu_q, mu_nq;
        Eigen::Matrix2d sigma_q, sigma_nq;
        mu_q = cell->getPointMean();
        sigma_q = cell->getPointCov();
        mu_nq = cell->getNormalMean();
        sigma_nq = cell->getNormalCov();
        Eigen::Vector3d g;
        Eigen::Matrix3d H;
        double score = 0;
        computeDerivatives(g, H, score, mu_p, sigma_p, mu_q, sigma_q, mu_np,
                           sigma_np, mu_nq, sigma_nq);
        scores += score;
        gs += g;
        Hs += H;
      }
    }
    return scores;
#else
      vector<NDTCell *> cells = targetNDT.getCellsForPoint(pt, neighbors, true);
      // dprintf("  (%ld/%ld): near %ld\n", i, nextNDT.size(), cells.size());
      for (size_t j = 0; j < cells.size(); ++j) {
        NDTCell *cell = cells.at(j);
        if (cell == NULL) {
          continue;
        }
        if (cell->phasGaussian_ && cell->nhasGaussian_) {
          Eigen::Vector2d mu_q, mu_nq;
          Eigen::Matrix2d sigma_q, sigma_nq;
          mu_q = cell->getPointMean();
          sigma_q = cell->getPointCov();
          mu_nq = cell->getNormalMean();
          sigma_nq = cell->getNormalCov();
          Eigen::Vector3d g;
          Eigen::Matrix3d H;
          double score = 0;
          computeDerivatives(g, H, score, mu_p, sigma_p, mu_q, sigma_q, mu_np,
                             sigma_np, mu_nq, sigma_nq);
          scores += score;
          gs += g;
          Hs += H;
        }
      }
    }
    return scores;
#endif
  }

 private:
  int itr_max;
  bool step_control;
  double delta_score;
  int neighbors;
  double d1, d2;
  double step_size;
  Eigen::Matrix<double, 2, 3> gm1;
  Eigen::Vector2d gm2, Hm1, Hm2;
  Eigen::Matrix2d gc1, gc2, Hc1, Hc2;

  void basicDerivatives(const Eigen::Vector2d &mu_p,
                        const Eigen::Matrix2d &sigma_p,
                        const Eigen::Vector2d &mu_np,
                        const Eigen::Matrix2d &sigma_np) {
    gm1 << 1, 0, -mu_p(1), 0, 1, mu_p(0);
    gm2 << -mu_np(1), mu_np(0);
    gc1 << -2 * sigma_p(0, 1), sigma_p(0, 0) - sigma_p(1, 1),
           sigma_p(0, 0) - sigma_p(1, 1), 2 * sigma_p(0, 1);
    gc2 << -2 * sigma_np(0, 1), sigma_np(0, 0) - sigma_np(1, 1),
           sigma_np(0, 0) - sigma_np(1, 1), 2 * sigma_np(0, 1);
    Hm1 << -mu_p(0), -mu_p(1);
    Hm2 << -mu_np(0), -mu_np(1);
    Hc1 << 2 * sigma_p(1, 1) - 2 * sigma_p(0, 0), -4 * sigma_p(0, 1),
           -4 * sigma_p(0, 1), 2 * sigma_p(0, 0) - 2 * sigma_p(1, 1);
    Hc2 << 2 * sigma_np(1, 1) - 2 * sigma_np(0, 0), -4 * sigma_np(0, 1),
           -4 * sigma_np(0, 1), 2 * sigma_np(0, 0) - 2 * sigma_np(1, 1);
  }

  bool computeDerivatives(Eigen::Vector3d &grad, Eigen::Matrix3d &hess,
                          double &score,
                          const Eigen::Vector2d &mu_p,
                          const Eigen::Matrix2d &sigma_p,
                          const Eigen::Vector2d &mu_q,
                          const Eigen::Matrix2d &sigma_q,
                          const Eigen::Vector2d &mu_np,
                          const Eigen::Matrix2d &sigma_np,
                          const Eigen::Vector2d &mu_nq,
                          const Eigen::Matrix2d &sigma_nq) {
    grad.setZero();
    hess.setZero();
    score = 0;
    Eigen::Vector2d m1 = mu_p - mu_q;
    Eigen::Matrix2d c1 = sigma_p + sigma_q;
    Eigen::Vector2d m2 = mu_np + mu_nq;
    Eigen::Matrix2d c2 = sigma_np + sigma_nq;
    Eigen::Vector3d gradF, gradG;
    Eigen::Matrix3d hessF, hessG;
    double trace1, trace2;
    traceDerivative(trace1, trace2, sigma_p, sigma_q, sigma_np, sigma_nq);
    computeFGDerivatives(gradF, gradG, hessF, hessG, m1, m2, c1, c2, trace1, trace2);
    double f = m1.dot(m2) * m1.dot(m2);
    double g = m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace();
    double l = f / g;
    if (isnan(l)) { return false; }
    score = -d1 * exp(-d2 * l / 2);
    grad = -d2 / (2 * g * g) * score * (gradF * g - f * gradG);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        if (i > j) {
          hess(i, j) = hess(j, i);
        } else {
          double Xpa = 1 / (g * g) * (gradF(i) * g - f * gradG(i));
          double Xpb = 1 / (g * g) * (gradF(j) * g - f * gradG(j));
          double Xp = Xpa * Xpb;
          double Yp =
              1 / (g * g) *
              (hessF(i, j) * g - gradF(i) * gradG(j) - gradF(j) * gradG(i) +
               2 * f / g * gradG(i) * gradG(j) - f * hessG(i, j));
          hess(i, j) = d2 / 2 * score * (d2 / 2 * Xp - Yp);
        }
      }
    }
    return true;
  }

  void computeFGDerivatives(Eigen::Vector3d &gradF, Eigen::Vector3d &gradG,
                            Eigen::Matrix3d &hessF, Eigen::Matrix3d &hessG,
                            const Eigen::Vector2d &m1,
                            const Eigen::Vector2d &m2,
                            const Eigen::Matrix2d &c1,
                            const Eigen::Matrix2d &c2,
                            double trace1, double trace2) {
    double m1m2 = m1.dot(m2);
    Eigen::Vector2d gm1x = gm1.col(0);
    Eigen::Vector2d gm1y = gm1.col(1);
    Eigen::Vector2d gm1t = gm1.col(2);
    double gm1m2t = gm1t.dot(m2) + m1.dot(gm2);
    gradF(0) = 2 * m1m2 * gm1x.dot(m2);
    gradF(1) = 2 * m1m2 * gm1y.dot(m2);
    gradF(2) = 2 * m1m2 * gm1m2t;

    gradG(0) = 2 * gm1x.dot(c2 * m1);
    gradG(1) = 2 * gm1y.dot(c2 * m1);
    gradG(2) = 2 * gm1t.dot(c2 * m1) + m1.dot(gc2 * m1) +
               2 * gm2.dot(c1 * m2) + m2.dot(gc1 * m2) + trace1;

    hessF(0, 0) = 2 * gm1x.dot(m2) * gm1x.dot(m2);
    hessF(0, 1) = 2 * gm1x.dot(m2) * gm1y.dot(m2);
    hessF(0, 2) = 2 * (gm1x.dot(m2) * gm1m2t + m1m2 * gm1x.dot(gm2));
    hessF(1, 1) = 2 * gm1y.dot(m2) * gm1y.dot(m2);
    hessF(1, 2) = 2 * (gm1y.dot(m2) * gm1m2t + m1m2 * gm1y.dot(gm2));
    double L = 2 * gm1t.dot(gm2) + Hm1.dot(m2) + Hm2.dot(m1);
    hessF(2, 2) = 2 * (gm1m2t * gm1m2t + m1m2 * L);
    hessF(1, 0) = hessF(0, 1);
    hessF(2, 0) = hessF(0, 2);
    hessF(2, 1) = hessF(1, 2);

    hessG(0, 0) = 2 * gm1x.dot(c2 * gm1x);
    hessG(0, 1) = 2 * gm1x.dot(c2 * gm1y);
    hessG(0, 2) = 2 * (gm1x.dot(gc2 * m1) + gm1x.dot(c2 * gm1t));
    hessG(1, 1) = 2 * gm1y.dot(c2 * gm1y);
    hessG(1, 2) = 2 * (gm1y.dot(gc2 * m1) + gm1y.dot(c2 * gm1t));
    hessG(2, 2) = 2 * Hm1.dot(c2 * m1) + m1.dot(Hc2 * m1) +
                  2 * (2 * gm1t.dot(gc2 * m1) + gm1t.dot(c2 * gm1t)) +
                  2 * Hm2.dot(c1 * m2) + m2.dot(Hc1 * m2) +
                  2 * (2 * gm2.dot(gc1 * m2) + gm2.dot(c1 * gm2)) + trace2;
    hessG(1, 0) = hessG(0, 1);
    hessG(2, 0) = hessG(0, 2);
    hessG(2, 1) = hessG(1, 2);
  }

  void traceDerivative(double &trace1, double &trace2,
                       const Eigen::Matrix2d &sigma_p,
                       const Eigen::Matrix2d &sigma_q,
                       const Eigen::Matrix2d &sigma_np,
                       const Eigen::Matrix2d &sigma_nq) {
    double ap = sigma_p(0, 0);
    double bp = sigma_p(0, 1);
    double cp = sigma_p(1, 1);
    double aq = sigma_q(0, 0);
    double bq = sigma_q(0, 1);
    double cq = sigma_q(1, 1);
    double anp = sigma_np(0, 0);
    double bnp = sigma_np(0, 1);
    double cnp = sigma_np(1, 1);
    double anq = sigma_nq(0, 0);
    double bnq = sigma_nq(0, 1);
    double cnq = sigma_nq(1, 1);
    trace1 = 2 * (anp * bq - anq * bp + ap * bnq - aq * bnp + bnp * cq -
                  bnq * cp + bp * cnq - bq * cnp);
    trace2 = -2 * anp * aq + 2 * anp * cq - 2 * anq * ap + 2 * anq * cp +
             2 * ap * cnq + 2 * aq * cnp - 8 * bnp * bq - 8 * bnq * bp -
             2 * cnp * cq - 2 * cnq * cp;
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
