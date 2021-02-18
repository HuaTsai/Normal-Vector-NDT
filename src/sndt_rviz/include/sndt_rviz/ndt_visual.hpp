#pragma once

#include <sndt/NDTCellMsg.h>
#include <sndt/NDTMapMsg.h>

namespace Ogre {
class Vector3;
class Quaternion;
class SceneManager;
class SceneNode;
}  // namespace Ogre

namespace rviz {
class Shape;
}

namespace sndt_rviz {

class NDTVisual {
 public:
  NDTVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node);
  virtual ~NDTVisual();
  void setCell(sndt::NDTCellMsg msg, double resolution);
  void setFramePosition(const Ogre::Vector3& position);
  void setFrameOrientation(const Ogre::Quaternion& orientation);
  void setColor(float r, float g, float b, float a);

 private:
  boost::shared_ptr<rviz::Shape> NDT_elipsoid_;
  Ogre::SceneNode* frame_node_;
  Ogre::SceneManager* scene_manager_;
};

}  // namespace sndt_rviz
