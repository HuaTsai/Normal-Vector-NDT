#pragma once

#include <Eigen/Dense>
#include <vector>

std::vector<Eigen::Vector2d> ComputeNormals(
    const std::vector<Eigen::Vector2d> &pc, double radius);
