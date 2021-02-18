#pragma once

#include <sndt/NDTMapMsg.h>
#include <rviz/message_filter_display.h>

#include <boost/circular_buffer.hpp>

namespace Ogre {
class SceneNode;
}

namespace rviz {
class ColorProperty;
class FloatProperty;
class IntProperty;
}  // namespace rviz

namespace sndt_rviz {

class NDTVisual;

class NDTDisplay : public rviz::MessageFilterDisplay<sndt::NDTMapMsg> {
  Q_OBJECT
 public:
  NDTDisplay();
  virtual ~NDTDisplay();

 protected:
  virtual void onInitialize();

  virtual void reset();

 private Q_SLOTS:
  void updateColorAndAlpha();
  void updateHistoryLength();

 private:
  /**
   * @brief transform a NDTmap msg into a vector of NDTVisual, one per cell. If
   * history is set to one or lower, it only display the last NDTMap. Otherwise,
   * it keeps all cells through time
   */
  void processMessage(const sndt::NDTMapMsg::ConstPtr& msg);

  std::vector<boost::shared_ptr<NDTVisual> > visuals_;

  rviz::ColorProperty* color_property_;
  rviz::FloatProperty* alpha_property_;
  rviz::IntProperty* history_length_property_;
};
}  // namespace sndt_rviz
