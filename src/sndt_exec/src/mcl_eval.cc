#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include "common/common.h"
#include <sndt/ndt_visualizations.h>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

/********** START OF RANDOM VECTOR ***********/
namespace Eigen {
namespace internal {
template <typename Scalar>
struct scalar_normal_dist_op {
  static boost::mt19937 rng;
  mutable boost::normal_distribution<Scalar> norm;
  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)
  template <typename Index>
  inline const Scalar operator()(Index, Index = 0) const {
    return norm(rng);
  }
};

template <typename Scalar>
boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

template <typename Scalar>
struct functor_traits<scalar_normal_dist_op<Scalar> > {
  enum {
    Cost = 50 * NumTraits<Scalar>::MulCost,
    PacketAccess = false,
    IsRepeatable = false
  };
};
}  // namespace internal
}  // namespace Eigen

Eigen::MatrixXd GenerateSamples(
    Eigen::VectorXd mean, Eigen::MatrixXd covar, int size, int nn,
    const Eigen::internal::scalar_normal_dist_op<double> &randN) {
  Eigen::MatrixXd normTransform(size, size);
  Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);
  if (cholSolver.info() == Eigen::Success) {
    normTransform = cholSolver.matrixL();
  } else {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    normTransform = eigenSolver.eigenvectors() *
                    eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }
  /* https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Computational_methods */
  Eigen::MatrixXd samples =
      (normTransform * Eigen::MatrixXd::NullaryExpr(size, nn, randN))
          .colwise() +
      mean;
  return samples;
}
/********** END OF RANDOM VECTOR ***********/


void StringToMeanAndCov(string str, Vector2d &mean, Matrix2d &cov) {
  regex sep(",");
  sregex_token_iterator pos(str.begin(), str.end(), sep, -1);
  mean(0) = stof(pos->str());
  mean(1) = stof((++pos)->str());
  cov(0, 0) = stof((++pos)->str());
  cov(0, 1) = cov(1, 0) = stof((++pos)->str());
  cov(1, 1) = stof((++pos)->str());
}

void StringToCenter(string str, Vector2d &cen) {
  regex sep(",");
  sregex_token_iterator pos(str.begin(), str.end(), sep, -1);
  cen(0) = stof(pos->str());
  cen(1) = stof((++pos)->str());
}

vector<double> ComputeMeanAndCov(
    const Eigen::Vector2d &p,
    const Eigen::Vector2d &q,
    const Eigen::Vector2d &np,
    const Eigen::Vector2d &nq,
    const Eigen::Matrix2d &pcov,
    const Eigen::Matrix2d &qcov,
    const Eigen::Matrix2d &npcov,
    const Eigen::Matrix2d &nqcov) {
  Eigen::Vector2d m1 = p - q;
  Eigen::Vector2d m2 = np + nq;
  Eigen::Matrix2d c1 = pcov + qcov;
  // Eigen::Matrix2d c2 = npcov + nqcov;
  vector<double> ret;
  ret.push_back(m1.dot(m2));
  // ret.push_back(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
  ret.push_back(m2.dot(c1 * m2));
  return ret;
}

int main(int argc, char **argv) {
  string pstr, qstr, npstr, nqstr, cpstr, cqstr;
  Vector2d pmean, npmean, qmean, nqmean, pcen, qcen;
  Matrix2d pcov, npcov, qcov, nqcov;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("p", po::value<string>(&pstr)->required(), "P Value")
      ("np", po::value<string>(&npstr)->required(), "NP Value")
      ("cp", po::value<string>(&cpstr)->required(), "P Cen Value")
      ("q", po::value<string>(&qstr)->required(), "Q Value")
      ("nq", po::value<string>(&nqstr)->required(), "NQ Value")
      ("cq", po::value<string>(&cqstr)->required(), "Q Cen Value");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
      cout << desc << endl;
      return 1;
  }
  StringToMeanAndCov(pstr, pmean, pcov);
  StringToMeanAndCov(npstr, npmean, npcov);
  StringToMeanAndCov(qstr, qmean, qcov);
  StringToMeanAndCov(nqstr, nqmean, nqcov);
  StringToCenter(cpstr, pcen);
  StringToCenter(cqstr, qcen);
  cout << " μp: (" << pmean(0) << ", " << pmean(1) << ")" << endl
       << " Σp: (" << pcov(0, 0) << ", " << pcov(0, 1) << ", "
                   << pcov(1, 0) << ", " << pcov(1, 1) << ")" << endl
       << "μnp: (" << npmean(0) << ", " << npmean(1) << ")" << endl
       << "Σnp: (" << npcov(0, 0) << ", " << npcov(0, 1) << ", "
                   << npcov(1, 0) << ", " << npcov(1, 1) << ")" << endl;
  cout << " μq: (" << qmean(0) << ", " << qmean(1) << ")" << endl
       << " Σq: (" << qcov(0, 0) << ", " << qcov(0, 1) << ", "
                   << qcov(1, 0) << ", " << qcov(1, 1) << ")" << endl
       << "μnq: (" << nqmean(0) << ", " << nqmean(1) << ")" << endl
       << "Σnq: (" << nqcov(0, 0) << ", " << nqcov(0, 1) << ", "
                   << nqcov(1, 0) << ", " << nqcov(1, 1) << ")" << endl;
  ros::init(argc, argv, "mcl_eval");
  ros::NodeHandle nh;
  NDTCell *cellp = new NDTCell();
  cellp->SetPointMean(pmean);
  cellp->SetPointCov(pcov);
  cellp->SetNormalMean(npmean);
  cellp->SetNormalCov(npcov);
  cellp->SetCenter(pcen);
  cellp->SetSize(1);
  cellp->SetNHasGaussian(true);

  NDTCell *cellq = new NDTCell();
  cellq->SetPointMean(qmean);
  cellq->SetPointCov(qcov);
  cellq->SetNormalMean(nqmean);
  cellq->SetNormalCov(nqcov);
  cellq->SetCenter(qcen);
  cellq->SetSize(1);
  cellq->SetNHasGaussian(true);

  Eigen::internal::scalar_normal_dist_op<double> randN;
  Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1);
  int nn = 10000;
  auto ps = GenerateSamples(pmean, pcov, 2, nn, randN);
  auto qs = GenerateSamples(qmean, qcov, 2, nn, randN);
  auto nps = GenerateSamples(npmean, npcov, 2, nn, randN);
  auto nqs = GenerateSamples(nqmean, nqcov, 2, nn, randN);
  vector<double> scores;
  double scsum = 0;
  for (int i = 0; i < nn; ++i) {
    // orignal sc
    // double sc = (ps.col(i) - qs.col(i)).dot(nps.col(i) + nqs.col(i));
    double sc = (ps.col(i) - qs.col(i)).dot(npmean + nqmean);
    scores.push_back(sc);
    scsum += sc;
  }
  double mean = scsum / nn;
  double sum = 0;
  for (auto &sc : scores)
    sum += (sc - mean) * (sc - mean);
  auto var = sum / nn;
  auto theory = ComputeMeanAndCov(pmean, qmean, npmean, nqmean, pcov, qcov, npcov, nqcov);
  cout << "practial: " << mean << ", " << var << endl;
  cout << "theory: " << theory[0] << ", " << theory[1] << endl;

  ros::Publisher pub1 = nh.advertise<visualization_msgs::MarkerArray>("markers1", 0, true);
  ros::Publisher pub2 = nh.advertise<visualization_msgs::MarkerArray>("markers2", 0, true);
  auto ma1 = MarkerArrayOfNDTCell(cellp);
  auto ma2 = MarkerArrayOfNDTCell2(cellq);
  auto ma3 = MarkerOfLines({cellp->GetPointMean(), cellq->GetPointMean()}, common::Color::kBlack, 1.0);
  pub1.publish(JoinMarkerArraysAndMarkers({ma1, ma2}, {ma3}));
  ros::spin();
}
