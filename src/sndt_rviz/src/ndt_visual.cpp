#include "sndt_rviz/ndt_visual.hpp"

#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreVector3.h>
#include <ros/ros.h>
#include <rviz/ogre_helpers/shape.h>

#include <Eigen/Dense>

namespace sndt_rviz {
NDTVisual::NDTVisual(Ogre::SceneManager* scene_manager,
                     Ogre::SceneNode* parent_node) {
  scene_manager_ = scene_manager;
  frame_node_ = parent_node->createChildSceneNode();
  NDT_elipsoid_.reset(
      new rviz::Shape(rviz::Shape::Sphere, scene_manager_, frame_node_));
}

NDTVisual::~NDTVisual() { scene_manager_->destroySceneNode(frame_node_); }

void NDTVisual::setCell(sndt::NDTCellMsg cell, double resolution) {
  Ogre::Vector3 position(cell.pmean[0], cell.pmean[1], 0);
  // eigen values aka size of the elipsoid
  Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      cov(i, j) = cell.pcov[i * 2 + j];
  Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
  Eigen::Matrix3d m_eigVal = es.pseudoEigenvalueMatrix();
  Eigen::Matrix3d m_eigVec = es.pseudoEigenvectors();
  m_eigVal = m_eigVal.cwiseSqrt();
  Eigen::Quaterniond q(m_eigVec);
  // Ogre::Vector3 scale(3*m_eigVal(0,0),3*m_eigVal(1,1),3*m_eigVal(2,2));
  Ogre::Vector3 scale(3 * m_eigVal(0, 0), 3 * m_eigVal(1, 1), 0.1);
  // fprintf(stderr, "scale %f, %f", m_eigVal(0, 0), m_eigVal(1, 1));
  Ogre::Quaternion orient(q.w(), q.x(), q.y(), q.z());

  // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> Sol (cov);
  // Eigen::Matrix3d evecs;
  // Eigen::Vector3d evals;
  // evecs = Sol.eigenvectors().real();
  // evals = Sol.eigenvalues().real();
  // Eigen::Quaterniond q(evecs);
  // Ogre::Vector3 scale(30*evals(0),30*evals(1),30*evals(2));
  // Ogre::Quaternion orient(q.w(),q.x(),q.y(),q.z());

  NDT_elipsoid_->setScale(scale);
  NDT_elipsoid_->setPosition(position);
  NDT_elipsoid_->setOrientation(orient);
}

void NDTVisual::setFramePosition(const Ogre::Vector3& position) {
  frame_node_->setPosition(position);
}

void NDTVisual::setFrameOrientation(const Ogre::Quaternion& orientation) {
  frame_node_->setOrientation(orientation);
}

void NDTVisual::setColor(float r, float g, float b, float a) {
  NDT_elipsoid_->setColor(r, g, b, a);
}
}  // namespace sndt_rviz
