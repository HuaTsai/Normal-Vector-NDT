// Created by Francois Gauthier-Clerc.
// Modified and refactored by HuaTsai.
#include <normal2d/normal2d.h>

void Normal2dEstimation::setInputCloud(const ConstPtrCloud& cloud) {
  m_in_cloud = cloud;
  m_indices->clear();
  m_indices->resize(cloud->points.size());
  for (unsigned int i = 0; i < cloud->points.size(); ++i) {
    (*m_indices)[i] = i;
  }
}

void Normal2dEstimation::setIndices(const pcl::PointIndices::Ptr& indices) {
  m_indices->clear();
  m_indices->resize(indices->indices.size());
  std::copy(indices->indices.cbegin(), indices->indices.cend(),
            m_indices->begin());
}

void Normal2dEstimation::setIndices(
    const pcl::PointIndices::ConstPtr& indices) {
  m_indices->clear();
  m_indices->resize(indices->indices.size());
  std::copy(indices->indices.cbegin(), indices->indices.cend(),
            m_indices->begin());
}

int Normal2dEstimation::searchForNeighbors(int index,
                                           std::vector<int>& nn_indices,
                                           std::vector<float>& nn_dists) const {
  nn_indices.clear();
  nn_dists.clear();
  if (m_k == 0) {
    m_kd_tree->radiusSearch(index, m_search_radius, nn_indices, nn_dists, 0);
  } else {
    m_kd_tree->nearestKSearch(index, m_k, nn_indices, nn_dists);
  }
  return nn_indices.size();
}

void Normal2dEstimation::compute(const PtrCloud& normal_cloud) const {
  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  boost::shared_ptr<std::vector<int>> nn_indices(new std::vector<int>(m_k));
  std::vector<float> nn_dists(m_k);

  normal_cloud->points.resize(m_in_cloud->points.size());
  normal_cloud->height = m_in_cloud->height;
  normal_cloud->width = m_in_cloud->width;

  if ((m_k == 0) && (m_search_radius == 0)) {
    throw std::runtime_error(
        "You must call once either setRadiusSearch or setKSearch !");
  }
  if ((m_k != 0) && (m_search_radius != 0)) {
    throw std::runtime_error(
        "You must call once either setRadiusSearch or setKSearch (not both) !");
  }

  m_kd_tree->setInputCloud(m_in_cloud, m_indices);

  normal_cloud->is_dense = true;
  // Save a few cycles by not checking every point for NaN/Inf values if the
  // cloud is set to dense
  if (m_in_cloud->is_dense) {
    // Iterating over the entire index vector
    for (unsigned int idx = 0; idx < m_indices->size(); ++idx) {
      if (searchForNeighbors((*m_indices)[idx], *nn_indices, nn_dists) == 0) {
        normal_cloud->points[idx].x = normal_cloud->points[idx].y =
            normal_cloud->points[idx].z =
                std::numeric_limits<float>::quiet_NaN();
        normal_cloud->is_dense = false;
        continue;
      }

      computePointNormal2d(nn_indices, normal_cloud->points[idx].x,
                           normal_cloud->points[idx].y,
                           normal_cloud->points[idx].z);
    }
  } else {
    // Iterating over the entire index vector
    for (unsigned int idx = 0; idx < m_indices->size(); ++idx) {
      if (!isFinite(m_in_cloud->points[(*m_indices)[idx]]) ||
          searchForNeighbors((*m_indices)[idx], *nn_indices, nn_dists) == 0) {
        normal_cloud->points[idx].x = normal_cloud->points[idx].y =
            normal_cloud->points[idx].z =
                std::numeric_limits<float>::quiet_NaN();
        normal_cloud->is_dense = false;
        continue;
      }

      computePointNormal2d(nn_indices, normal_cloud->points[idx].x,
                           normal_cloud->points[idx].y,
                           normal_cloud->points[idx].z);
    }
  }
}

void Normal2dEstimation::compute(
    const pcl::PointCloud<pcl::Normal>::Ptr& normal_cloud) const {
  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  boost::shared_ptr<std::vector<int>> nn_indices(new std::vector<int>(m_k));
  std::vector<float> nn_dists(m_k);

  normal_cloud->points.resize(m_in_cloud->points.size());
  normal_cloud->height = m_in_cloud->height;
  normal_cloud->width = m_in_cloud->width;

  if ((m_k == 0) && (m_search_radius == 0)) {
    throw std::runtime_error(
        "You must call once either setRadiusSearch or setKSearch !");
  }

  if ((m_k != 0) && (m_search_radius != 0)) {
    throw std::runtime_error(
        "You must call once either setRadiusSearch or setKSearch (not both) !");
  }

  m_kd_tree->setInputCloud(m_in_cloud, m_indices);

  normal_cloud->is_dense = true;
  // Save a few cycles by not checking every point for NaN/Inf values if the
  // cloud is set to dense
  if (m_in_cloud->is_dense) {
    // Iterating over the entire index vector
    for (unsigned int idx = 0; idx < m_indices->size(); ++idx) {
      if (searchForNeighbors((*m_indices)[idx], *nn_indices, nn_dists) == 0) {
        normal_cloud->points[idx].normal_x =
            normal_cloud->points[idx].normal_y =
                normal_cloud->points[idx].normal_z =
                    std::numeric_limits<float>::quiet_NaN();
        normal_cloud->is_dense = false;
        continue;
      }

      computePointNormal2d(nn_indices, normal_cloud->points[idx].normal_x,
                           normal_cloud->points[idx].normal_y,
                           normal_cloud->points[idx].normal_z,
                           normal_cloud->points[idx].curvature);
    }
  } else {
    // Iterating over the entire index vector
    for (unsigned int idx = 0; idx < m_indices->size(); ++idx) {
      if (!isFinite(m_in_cloud->points[(*m_indices)[idx]]) ||
          searchForNeighbors((*m_indices)[idx], *nn_indices, nn_dists) == 0) {
        normal_cloud->points[idx].normal_x =
            normal_cloud->points[idx].normal_y =
                normal_cloud->points[idx].normal_z =
                    std::numeric_limits<float>::quiet_NaN();

        normal_cloud->is_dense = false;
        continue;
      }

      computePointNormal2d(nn_indices, normal_cloud->points[idx].normal_x,
                           normal_cloud->points[idx].normal_y,
                           normal_cloud->points[idx].normal_z,
                           normal_cloud->points[idx].curvature);
    }
  }
}

bool Normal2dEstimation::computePointNormal2d(
    boost::shared_ptr<std::vector<int>>& indices, float& nx, float& ny,
    float& nz) const {
  if (indices->size() < 2) {
    nx = ny = nz = std::numeric_limits<float>::quiet_NaN();
    return false;
  }
  if (indices->size() == 2) {
    double norm, vect_x, vect_y;
    vect_x = m_in_cloud->points[(*indices)[0]].x -
             m_in_cloud->points[(*indices)[1]].x;
    vect_y = m_in_cloud->points[(*indices)[0]].y -
             m_in_cloud->points[(*indices)[1]].y;
    norm = std::pow(std::pow(vect_x, 2.0) + std::pow(vect_y, 2.0), 0.5);
    vect_x /= norm;
    vect_y /= norm;
    nx = -vect_y;
    ny = vect_x;
  } else {
    // Get the plane normal and surface curvature
    PCA2D pca;
    pca.setInputCloud(m_in_cloud);
    pca.setIndices(indices);
    // Note that declare auto in release mode will not work
    Eigen::Vector2f result = pca.getEigenVectors().col(1);
    nx = result(0);
    ny = result(1);
  }

  if (ny < 0) {
    nx = -nx;
    ny = -ny;
  }
  nz = 0.0;
  return true;
}

bool Normal2dEstimation::computePointNormal2d(
    boost::shared_ptr<std::vector<int>>& indices, float& nx, float& ny,
    float& nz, float& curvature) const {
  if (indices->size() < 2) {
    nx = ny = nz = curvature = std::numeric_limits<float>::quiet_NaN();
    return false;
  }
  if (indices->size() == 2) {
    double norm, vect_x, vect_y;
    vect_x = m_in_cloud->points[(*indices)[0]].x -
             m_in_cloud->points[(*indices)[1]].x;
    vect_y = m_in_cloud->points[(*indices)[0]].y -
             m_in_cloud->points[(*indices)[1]].y;
    norm = std::pow(std::pow(vect_x, 2.0) + std::pow(vect_y, 2.0), 0.5);
    vect_x /= norm;
    vect_y /= norm;
    nx = -vect_y;
    ny = vect_x;
    curvature = 0.0;
  } else {
    // Get the plane normal and surface curvature
    PCA2D pca;
    pca.setInputCloud(m_in_cloud);
    pca.setIndices(indices);
    // Note that declare auto in release mode will not work
    Eigen::Vector2f result = pca.getEigenVectors().col(1);
    nx = result(0);
    ny = result(1);
    curvature = pca.getEigenValues()(1) /
                (pca.getEigenValues()(0) + pca.getEigenValues()(1));
  }

  if (ny < 0) {
    nx = -nx;
    ny = -ny;
  }
  nz = 0.0;
  return true;
}
