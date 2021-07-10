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
#include <pcl/filters/filter.h>
#include <TriMesh.h>
#include <ICP.h>
#include <pcl/registration/transformation_estimation_symmetric_point_to_plane_lls.h>
#include "normal2d/Normal2dEstimation.hpp"
#include "common/common.h"

using namespace std;
using namespace trimesh;
using namespace Eigen;

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

pcl::PointCloud<pcl::PointXYZ> PCLFromMatrixXd(const Eigen::MatrixXd &mtx) {
  pcl::PointCloud<pcl::PointXYZ> ret;
  for (int i = 0; i < mtx.cols(); ++i) {
    ret.push_back(pcl::PointXYZ(mtx(0, i), mtx(1, i), mtx(2, i)));
  }
  return ret;
}

Eigen::MatrixXd MatrixXdFromPCL(const pcl::PointCloud<pcl::PointXYZ> &pc) {
  Eigen::MatrixXd ret;
  for (const auto &pt : pc.points) {
    ret.conservativeResize(3, ret.cols() + 1);
    ret.col(ret.cols() - 1) = Eigen::Vector3d(pt.x, pt.y, pt.z);
  }
  return ret;
}

Eigen::MatrixXd MatrixXdFromPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr pc) {
  Eigen::MatrixXd ret;
  for (const auto &pt : pc->points) {
    ret.conservativeResize(3, ret.cols() + 1);
    ret.col(ret.cols() - 1) = Eigen::Vector3d(pt.x, pt.y, pt.z);
  }
  return ret;
}

template <typename PointCloudPtrType>
void ComputeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                    PointCloudPtrType normals) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(normals);
}

void ComputeNormalsAndJoin(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                           pcl::PointCloud<pcl::PointNormal>::Ptr output) {
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ComputeNormals(pc, radius, normals);
  pcl::concatenateFields(*pc, *normals, *output);
}

void RemoveNaN(pcl::PointCloud<pcl::PointNormal>::Ptr pc) {
  auto pts = pc->points;
  pc->clear();
  for (const auto &pt : pts) {
    if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z) && !isnan(pt.normal_x) &&
        !isnan(pt.normal_y) && !isnan(pt.normal_z)) {
      pc->push_back(pt);
    }
  }
}

void RemoveNaNAndInf(pcl::PointCloud<pcl::PointNormal>::Ptr pc) {
  auto pts = pc->points;
  pc->clear();
  for (const auto &pt : pts)
    if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z) &&
        isfinite(pt.normal_x) && isfinite(pt.normal_y) && isfinite(pt.normal_z))
      pc->push_back(pt);
}

Eigen::Matrix3d Matrix3dFromXForm(const xform &xf) {
  Eigen::Matrix4f mtx4f;
  mtx4f << xf(0, 0), xf(0, 1), xf(0, 2), xf(0, 3),
           xf(1, 0), xf(1, 1), xf(1, 2), xf(1, 3),
           xf(2, 0), xf(2, 1), xf(2, 2), xf(2, 3),
           xf(3, 0), xf(3, 1), xf(3, 2), xf(3, 3);
  return common::Matrix3dFromMatrix4f(mtx4f);
}

void DoSICP(common::MatchPackage &mp, const vector<double> &params) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr mpsource(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr mptarget(new pcl::PointCloud<pcl::PointXYZ>);
  *mpsource = PCLFromMatrixXd(mp.source);
  *mptarget = PCLFromMatrixXd(mp.target);

  pcl::PointCloud<pcl::PointNormal>::Ptr source(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr target(new pcl::PointCloud<pcl::PointNormal>);
  ComputeNormalsAndJoin(mpsource, 1, source);
  ComputeNormalsAndJoin(mptarget, 1, target);
  RemoveNaN(source);
  RemoveNaN(target);

  TriMesh *mesh1 = new TriMesh();
  TriMesh *mesh2 = new TriMesh();

  for (const auto &pt : *target) {
    mesh1->vertices.push_back(point(pt.x, pt.y, pt.z));
    mesh1->normals.push_back(point(pt.normal_x, pt.normal_y, pt.normal_z));
  }
  for (const auto &pt : *source) {
    mesh2->vertices.push_back(point(pt.x, pt.y, pt.z));
    mesh2->normals.push_back(point(pt.normal_x, pt.normal_y, pt.normal_z));
  }

  KDtree *kd1 = new KDtree(mesh1->vertices);
	KDtree *kd2 = new KDtree(mesh2->vertices);

  xform xf1;
  xform xf2 = xform(common::Matrix4fFromMatrix3d(mp.guess).data());

  vector<float> weights1, weights2;
  ICP(mesh1, mesh2, xf1, xf2, kd1, kd2, weights1, weights2, 0.0f, 2, ICP_RIGID);

  // mp.state = ???
  // mp.iters = ???
  mp.iters = iters;
  mp.result = Matrix3dFromXForm(xf2);

  free(mesh1);
  free(mesh2);
  free(kd1);
  free(kd2);
}

void DoPCLSICP(common::MatchPackage &mp, const vector<double> &params) {
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> matcher;
  matcher.setUseSymmetricObjective(true);

  matcher.setMaximumIterations(static_cast<int>(params.at(0)));
  matcher.setMaxCorrespondenceDistance(params.at(1));
  matcher.setUseReciprocalCorrespondences((params.at(2) != 0) ? true : false);
  matcher.setEuclideanFitnessEpsilon(params.at(3));
  matcher.setTransformationEpsilon(params.at(4));

  pcl::PointCloud<pcl::PointXYZ>::Ptr mpsource(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr mptarget(new pcl::PointCloud<pcl::PointXYZ>);
  *mpsource = PCLFromMatrixXd(mp.source);
  *mptarget = PCLFromMatrixXd(mp.target);

  pcl::PointCloud<pcl::PointNormal>::Ptr source(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr target(new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr output(new pcl::PointCloud<pcl::PointNormal>);
  ComputeNormalsAndJoin(mpsource, 1, source);
  ComputeNormalsAndJoin(mptarget, 1, target);
  // RemoveNaN(source);
  // RemoveNaN(target);

  matcher.setInputSource(source);
  matcher.setInputTarget(target);
  matcher.align(*output, common::Matrix4fFromMatrix3d(mp.guess));

  // mp.state = FromPCLState(matcher.convergence_criteria_->getConvergenceState());
  mp.iters = matcher.nr_iterations_;
  mp.result = common::Matrix3dFromMatrix4f(matcher.getFinalTransformation());
}

/**
 * @param data 
 * @param params (radius)
 */
TriMesh *MakeMesh(const MatrixXd &data,
                  const vector<double> &params) {
  double radius = params[0];
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < data.cols(); ++i)
    if (data.col(i).allFinite())
      pc->push_back(pcl::PointXYZ(data.col(i)(0), data.col(i)(1), 0));
  pcl::PointCloud<pcl::PointNormal>::Ptr pcn(new pcl::PointCloud<pcl::PointNormal>);
  ComputeNormalsAndJoin(pc, radius, pcn);
  RemoveNaNAndInf(pcn);
  TriMesh *ret = new TriMesh();
  for (const auto &pt : *pcn) {
    ret->vertices.push_back(point(pt.x, pt.y, pt.z));
    ret->normals.push_back(point(pt.normal_x, pt.normal_y, pt.normal_z));
  }
  return ret;
}
