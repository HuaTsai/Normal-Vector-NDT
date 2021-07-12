#pragma once

#include <bits/stdc++.h>
#include <common/common.h>

#include <Eigen/Eigen>

#include "sndt/ndt_map.h"

using namespace std;
using namespace Eigen;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

vector<Vector2d> FindTangentPoints(const Marker &eclipse,
                                   const Vector2d &point);

void UpdateMarkerArray(MarkerArray &markerarray, Marker marker);

MarkerArray JoinMarkers(const vector<Marker> &ms);

MarkerArray JoinMarkerArraysAndMarkers(const vector<MarkerArray> &mas,
                                       const vector<Marker> &ms = {});

Marker MarkerOfBoundary(const Vector2d &center, double size,
                        double skew_rad = 0,
                        const common::Color &color = common::Color::kBlack);

Marker MarkerOfEclipse(const Vector2d &mean, const Matrix2d &covariance,
                       const common::Color &color = common::Color::kLime,
                       double alpha = 0.6);

Marker MarkerOfLines(const vector<Vector2d> &points,
                     const common::Color &color = common::Color::kGray,
                     double alpha = 0.6);

Marker MarkerOfLinesByEndPoints(
    const vector<Vector2d> &points,
    const common::Color &color = common::Color::kGray, double alpha = 0.6);

vector<Vector2d> PointsOfNDTMap(const NDTMap &map);

vector<Vector2d> PointsOfNDTMap(const vector<shared_ptr<NDTCell>> &map);

Marker MarkerOfPoints(const vector<Vector2d> &points, double size = 0.1,
                      const common::Color &color = common::Color::kLime,
                      double alpha = 1.0);

Marker MarkerOfPoints(const MatrixXd &points, double size = 0.1,
                      const common::Color &color = common::Color::kLime,
                      double alpha = 1.0);

Marker MarkerOfArrow(const Vector2d &start, const Vector2d &end,
                     const common::Color &color = common::Color::kGray,
                     double alpha = 0.6);

MarkerArray MarkerArrayOfArrow(const MatrixXd &start, const MatrixXd &end,
                               const common::Color &color = common::Color::kRed,
                               double alpha = 1.0);

// Color set for general usage
MarkerArray MarkerArrayOfNDTCell(const NDTCell *cell);

// Color set for target point cloud
MarkerArray MarkerArrayOfNDTCell2(const NDTCell *cell);

MarkerArray MarkerArrayOfNDTMap(const NDTMap &map,
                                bool is_target_color = false);

MarkerArray MarkerArrayOfNDTMap(const vector<shared_ptr<NDTCell>> &map,
                                bool is_target_color = false);

MarkerArray MarkerArrayOfSensor(const vector<Affine2d> &affs);
