#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/cost_functors.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <std_msgs/Int32.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.h>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;
const auto &Avg = Average;

double Stdev(const vector<double> &c, double m) {
  return accumulate(c.begin(), c.end(), 0.,
                    [&m](auto a, auto b) { return a + (b - m) * (b - m); }) /
         (c.size() - 1);
}

Vector4d ComputeMean(const vector<Vector4d> &points) {
  Eigen::Vector4d ret = Eigen::Vector4d::Zero();
  ret = accumulate(points.begin(), points.end(), ret);
  ret /= points.size();
  return ret;
}

Matrix4d ComputeCov(const std::vector<Eigen::Vector4d> &points,
                    const Eigen::Vector4d &mean) {
  int n = points.size();
  Eigen::MatrixXd mp(4, n);
  for (int i = 0; i < n; ++i) mp.col(i) = points.at(i) - mean;
  Eigen::Matrix4d ret;
  ret = mp * mp.transpose() / (n - 1);
  return ret;
}

int main() {
  Vector2d up, uq, unp, unq;
  Matrix2d cp, cq, cnp, cnq;
  up << 1.97662, -15.4599;
  cp << 0.0501161, -0.0518003, -0.0518003, 0.268278;
  unp << 0.961995, -0.00630887;
  cnp << 0.0010206, 0.00836658, 0.00836658, 0.0884107;
  uq << 4.19542, -14.8495;
  cq << 0.0204102, -0.00978165, -0.00978165, 0.266181;
  unq << 0.995777, 0.0097305;
  cnq << 1.97988e-05, 0.000276823, 0.000276823, 0.0110923;

  int n = 100000;
  auto ps = GenerateGaussianSamples(up, cp, n);
  auto qs = GenerateGaussianSamples(uq, cq, n);
  auto nps = GenerateGaussianSamples(unp, cnp, n);
  auto nqs = GenerateGaussianSamples(unq, cnq, n);
  // vector<double> costs;
  // for (int i = 0; i < n; ++i) {
  //   costs.push_back((ps[i] - qs[i]).dot(nps[i] + nqs[i]));
  // }
  // double mean = Avg(costs);
  // double stdev = Stdev(costs, mean);
  // cout << mean << ", " << stdev << " -> " << mean / stdev << endl;

  // Vector2d m1 = up - uq;
  // Vector2d m2 = unp + unq;
  // Matrix2d c1 = cp + cq;
  // Matrix2d c2 = cnp + cnq;
  // double num = m1.dot(m2);
  // double den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
  // cout << num << ", " << den << " -> " << num / den << endl;

  vector<Vector2d> m1s;
  vector<Vector4d> pqs;
  for (int i = 0; i < n; ++i) {
    m1s.push_back((ps[i] - qs[i]));
    pqs.push_back(Vector4d(ps[i](0), ps[i](1), qs[i](0), qs[i](1)));
  }
  cout << ComputeMean(ps).transpose() << endl;
  cout << ComputeCov(ps, ComputeMean(ps)) << endl;
  cout << ComputeMean(pqs).transpose() << endl;
  cout << ComputeCov(pqs, ComputeMean(pqs)) << endl;
  auto mean2 = ComputeMean(m1s);
  auto cov2 = ComputeCov(m1s, mean2);
  // cout << "cp + cq" << endl << c1 << endl;
  cout << "ac cov" << endl << cov2 << endl;
  // cout << "cp: " << endl << cp << endl;
  // cout << "ac cp: " << endl << ComputeCov(ps, ComputeMean(ps)) << endl;
  // cout << "cq: " << endl << cq << endl;
  // cout << "ac cq: " << endl << ComputeCov(qs, ComputeMean(qs)) << endl;
  MatrixXd C(4, 4);
  C.setZero();
  C.block<2, 2>(0, 0) = cp;
  C.block<2, 2>(2, 2) = cq;
  MatrixXd R(2, 4);
  R << 1, 0, -1, 0, 0, 1, 0, -1;
  cout << "R * C * R^T: " << endl << R * C * R.transpose() << endl;
  cout << "R * C * R^T: " << endl
       << R * ComputeCov(pqs, ComputeMean(pqs)) * R.transpose() << endl;
}
// cost: -5.11832
// 0.5 * c * c: 13.0986