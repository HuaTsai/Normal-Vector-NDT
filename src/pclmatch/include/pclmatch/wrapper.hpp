#pragma once

#include <bits/stdc++.h>
#include <pcl/pcl_config.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/ndt_2d.h>
#include <pcl/registration/transformation_estimation_2D.h>

#if (PCL_MINOR_VERSION == 11)
#include <pcl/registration/transformation_estimation_symmetric_point_to_plane_lls.h>
#include "normal2d/Normal2dEstimation.hpp"
#endif

using namespace std;

typedef pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState State;

enum ConvergeState {
  NOT_CONVERGED,
  ITERATIONS,
  TRANSFORM,
  ABS_MSE,
  REL_MSE,
  NO_CORRESPONDENCES,
  FAILURE_AFTER_MAX_ITERATIONS
};

enum ConvergeState FromPCLState(State state) {
  ConvergeState ret;
  if (state == State::CONVERGENCE_CRITERIA_NOT_CONVERGED) {
    ret = ConvergeState::NOT_CONVERGED;
  } else if (state == State::CONVERGENCE_CRITERIA_ITERATIONS) {
    ret = ConvergeState::ITERATIONS;
  } else if (state == State::CONVERGENCE_CRITERIA_TRANSFORM) {
    ret = ConvergeState::TRANSFORM;
  } else if (state == State::CONVERGENCE_CRITERIA_ABS_MSE) {
    ret = ConvergeState::ABS_MSE;
  } else if (state == State::CONVERGENCE_CRITERIA_REL_MSE) {
    ret = ConvergeState::REL_MSE;
  } else if (state == State::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES) {
    ret = ConvergeState::NO_CORRESPONDENCES;
  } else if (state == State::CONVERGENCE_CRITERIA_FAILURE_AFTER_MAX_ITERATIONS) {
    ret = ConvergeState::FAILURE_AFTER_MAX_ITERATIONS;
  }
  return ret;
}

struct MatchConfig {
  string reg;
  int max_iters;
  double max_corr_dists;
  bool set_reci_corr;
  double l2_rel_eps;
  double tf_eps;
  // SICP
  double neighbors;
  // NDT
  double cell_size;
};

struct MatchPackage {
  MatchPackage()
      : source(new pcl::PointCloud<pcl::PointXYZ>),
        target(new pcl::PointCloud<pcl::PointXYZ>),
        output(new pcl::PointCloud<pcl::PointXYZ>),
        guess(Eigen::Matrix3d::Identity()),
        result(Eigen::Matrix3d::Identity()),
        actual(Eigen::Matrix3d::Identity()),
        iters(0),
        state(NOT_CONVERGED) {}
  pcl::PointCloud<pcl::PointXYZ>::Ptr source;
  pcl::PointCloud<pcl::PointXYZ>::Ptr target;
  pcl::PointCloud<pcl::PointXYZ>::Ptr output;
  Eigen::Matrix3d guess;
  Eigen::Matrix3d result;
  Eigen::Matrix3d actual;
  int iters;
  enum ConvergeState state;
};

class MatchingBatch2D {
 public:
  
 private:
  Eigen::MatrixXd source_;
  Eigen::MatrixXd target_;
  std::vector<Eigen::Matrix3d> guesses_;
  std::vector<Eigen::Matrix3d> results_;
};

void ComputeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                    pcl::PointCloud<pcl::PointNormal>::Ptr output) {
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(normals);
  pcl::concatenateFields(*pc, *normals, *output);
}

void PrintVersion(string str) {
  cout << str << " @ PCL " << PCL_VERSION_PRETTY << endl;
}

void DoICP(MatchPackage &mp, const vector<double> &params) {
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> matcher;

  matcher.setMaximumIterations(static_cast<int>(params.at(0)));
  matcher.setMaxCorrespondenceDistance(params.at(1));
  matcher.setUseReciprocalCorrespondences((params.at(2) != 0) ? true : false);
  matcher.setEuclideanFitnessEpsilon(params.at(3));
  matcher.setTransformationEpsilon(params.at(4));

  matcher.setInputSource(mp.source);
  matcher.setInputTarget(mp.target);
  matcher.align(*mp.output, common::Matrix4fFromMatrix3d(mp.guess));

  mp.state = FromPCLState(matcher.convergence_criteria_->getConvergenceState());
  mp.iters = matcher.nr_iterations_;
  mp.result = common::Matrix3dFromMatrix4f(matcher.getFinalTransformation());
}

void DoICPNL(MatchPackage &mp, const vector<double> &params) {
  pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> matcher;

  matcher.setMaximumIterations(static_cast<int>(params.at(0)));
  matcher.setMaxCorrespondenceDistance(params.at(1));
  matcher.setUseReciprocalCorrespondences((params.at(2) != 0) ? true : false);
  matcher.setEuclideanFitnessEpsilon(params.at(3));
  matcher.setTransformationEpsilon(params.at(4));

  matcher.setInputSource(mp.source);
  matcher.setInputTarget(mp.target);
  matcher.align(*mp.output, common::Matrix4fFromMatrix3d(mp.guess));

  mp.state = FromPCLState(matcher.convergence_criteria_->getConvergenceState());
  mp.iters = matcher.nr_iterations_;
  mp.result = common::Matrix3dFromMatrix4f(matcher.getFinalTransformation());
}

void DoGICP(MatchPackage &mp, const vector<double> &params) {
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> matcher;

  matcher.setMaximumIterations(static_cast<int>(params.at(0)));
  matcher.setMaxCorrespondenceDistance(params.at(1));
  matcher.setUseReciprocalCorrespondences((params.at(2) != 0) ? true : false);
  matcher.setEuclideanFitnessEpsilon(params.at(3));
  matcher.setTransformationEpsilon(params.at(4));

  matcher.setInputSource(mp.source);
  matcher.setInputTarget(mp.target);
  matcher.align(*mp.output, common::Matrix4fFromMatrix3d(mp.guess));

  mp.state = FromPCLState(matcher.convergence_criteria_->getConvergenceState());
  mp.iters = matcher.nr_iterations_;
  mp.result = common::Matrix3dFromMatrix4f(matcher.getFinalTransformation());

}

// void DoICPPt2Pl(MatchPackage &mp, const vector<double> &params) {
//   pcl::IterativeClosestPointWithNormals<pcl::PointXYZ, pcl::PointXYZ> matcher;
//   matcher.setInputSource(mp.source);
//   matcher.setInputTarget(mp.target);
//   matcher.align(*mp.output);
// }

// void DoNDTP2D(MatchPackage &mp, const vector<double> &params) {
//   pcl::NormalDistributionsTransform2D<pcl::PointXYZ, pcl::PointXYZ> matcher;
//   matcher.setInputSource(mp.source);
//   matcher.setInputTarget(mp.target);
//   matcher.align(*mp.output);
// }

#if (PCL_MINOR_VERSION == 11)
void DoSICP(MatchPackage &mp, const vector<double> &params) {
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> matcher;
  matcher.setUseSymmetricObjective(true);

  matcher.setMaximumIterations(static_cast<int>(params.at(0)));
  matcher.setMaxCorrespondenceDistance(params.at(1));
  matcher.setUseReciprocalCorrespondences((params.at(2) != 0) ? true : false);
  matcher.setEuclideanFitnessEpsilon(params.at(3));
  matcher.setTransformationEpsilon(params.at(4));

  pcl::PointCloud<pcl::PointNormal>::Ptr source(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr target(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr output(new pcl::PointCloud<pcl::PointNormal>);
  ComputeNormals(mp.source, 1, source);
  ComputeNormals(mp.target, 1, target);
  matcher.setInputSource(source);
  matcher.setInputTarget(target);
  matcher.align(*output, common::Matrix4fFromMatrix3d(mp.guess));

  mp.state = FromPCLState(matcher.convergence_criteria_->getConvergenceState());
  mp.iters = matcher.nr_iterations_;
  mp.result = common::Matrix3dFromMatrix4f(matcher.getFinalTransformation());
}
#endif
