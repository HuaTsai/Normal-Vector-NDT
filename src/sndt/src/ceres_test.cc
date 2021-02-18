#include <bits/stdc++.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

Eigen::Matrix3d GenTF(double x, double y, double th) {
  Eigen::Matrix3d ret;
  double rad = th * M_PI / 180.;
  ret << cos(rad), -sin(rad), x, sin(rad), cos(rad), y, 0, 0, 1;
  return ret;
}

struct Cell {
  Cell() {
    mu.setZero();
    mu_n.setZero();
    sigma.setZero();
    sigma_n.setZero();
  }
  explicit Cell(const Eigen::Vector2d &mu_, const Eigen::Matrix2d &sigma_,
                const Eigen::Vector2d &mu_n_, const Eigen::Matrix2d &sigma_n_)
      : mu(mu_), mu_n(mu_n_), sigma(sigma_), sigma_n(sigma_n_) {}
  Eigen::Vector2d mu, mu_n;
  Eigen::Matrix2d sigma, sigma_n;
};

struct CostFunction {
  const Cell p_, q_;
  CostFunction(Cell p, Cell q) : p_(p), q_(q) {}
  template<typename T>
  bool operator()(const T *const xyt, T *e) const {
    T th = xyt[2] * M_PI / 180.;
    T m1[2] = {cos(th) * p_.mu(0) - sin(th) * p_.mu(1) + xyt[0] - q_.mu(0),
               sin(th) * p_.mu(0) + cos(th) * p_.mu(1) + xyt[1] - q_.mu(1)};
    T m2[2] = {cos(th) * p_.mu_n(0) - sin(th) * p_.mu_n(1) + q_.mu_n(0),
               sin(th) * p_.mu_n(0) + cos(th) * p_.mu_n(1) + q_.mu_n(1)};
    double ap = p_.sigma(0, 0);
    double bp = p_.sigma(0, 1);
    double cp = p_.sigma(1, 1);
    double anp = p_.sigma_n(0, 0);
    double bnp = p_.sigma_n(0, 1);
    double cnp = p_.sigma_n(1, 1);
    double aq = q_.sigma(0, 0);
    double bq = q_.sigma(0, 1);
    double cq = q_.sigma(1, 1);
    double anq = q_.sigma_n(0, 0);
    double bnq = q_.sigma_n(0, 1);
    double cnq = q_.sigma_n(1, 1);
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
    e[0] = -l;
    return true;
  }
};

void GenData(double x, double y, double th,
             Cell &cell1, Cell &cell2) {
  cell1.mu << 22, -4.04059;
  cell1.sigma << 1.1, -0.166179, -0.166179, 0.224476;
  cell1.mu_n << 0.077847, 0.797986;
  cell1.sigma_n << 0.607864, -0.0126734, -0.0126734, 0.127874;

  Eigen::Matrix3d T = GenTF(x, y, th);
  Eigen::Matrix2d R = T.block<2, 2>(0, 0);
  Eigen::Vector2d t = T.block<2, 1>(0, 2);

  cell2.mu = R * cell1.mu + t;
  cell2.mu_n = R * cell1.mu_n;
  cell2.sigma = R * cell1.sigma * R.transpose();
  cell2.sigma_n = R * cell1.sigma_n * R.transpose();
}

void ComputeScore(const Cell &p, const Cell &q, const vector<double> &xyt) {
  double th = xyt[2] * M_PI / 180.;
  Eigen::Matrix2d R;
  R << cos(th), -sin(th), sin(th), cos(th);
  Eigen::Vector2d t(xyt[0], xyt[1]);
  Eigen::Vector2d m1 = R * p.mu + t - q.mu;
  Eigen::Vector2d m2 = R * p.mu_n + q.mu_n;
  Eigen::Matrix2d c1 = R * p.sigma * R.transpose() + q.sigma;
  Eigen::Matrix2d c2 = R * p.sigma_n * R.transpose() + q.sigma_n;
  double f = m1.dot(m2) * m1.dot(m2);
  double g = m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace();
  double l = f / g;
  double score = -exp(-l / 2);
  printf("score @ (%.2f, %.2f, %.2f) = %f, %f\n", xyt[0], xyt[1], xyt[2], score, l);
}

int main(int argc, char* argv[]) {
  Cell p, q;
  GenData(5, 5, 10, p, q);

  ceres::Problem problem;
  double xyt[3] = {0};
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
                               new CostFunction(p, q)), nullptr, xyt);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  cout << summary.BriefReport() << endl;
  printf("est (x, y, t) = (%.2f, %.2f, %.2f)\n", xyt[0], xyt[1], xyt[2]);
  printf("cost %f -> %f\n", summary.initial_cost, summary.final_cost);
  ComputeScore(p, q, {xyt[0], xyt[1], xyt[2]});
  ComputeScore(p, q, {0, 0, 0});
  ComputeScore(p, q, {4.5, 4.5, 8});
  ComputeScore(p, q, {5, 5, 10});
}
