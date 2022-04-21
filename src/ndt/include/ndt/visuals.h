#include <ndt/nmap.h>
#include <ndt/nmap2d.h>
#include <visualization_msgs/MarkerArray.h>

enum class MarkerOptions { kRed, kGreen, kCell, kCov, kNormal };

visualization_msgs::MarkerArray MarkerOfNDT(
    const NMap &map,
    const std::unordered_set<MarkerOptions> &options,
    const Eigen::Affine3d &T = Eigen::Affine3d::Identity());

visualization_msgs::MarkerArray MarkerOfNDT(
    const std::shared_ptr<NMap> &map,
    const std::unordered_set<MarkerOptions> &options,
    const Eigen::Affine3d &T = Eigen::Affine3d::Identity());

visualization_msgs::MarkerArray MarkerOfNDT(
    const NMap2D &map,
    const std::unordered_set<MarkerOptions> &options,
    const Eigen::Affine3d &T = Eigen::Affine3d::Identity());

visualization_msgs::MarkerArray MarkerOfNDT(
    const std::shared_ptr<NMap2D> &map,
    const std::unordered_set<MarkerOptions> &options,
    const Eigen::Affine2d &T = Eigen::Affine2d::Identity());

visualization_msgs::MarkerArray MarkerOfCell(
    const Cell &cell, const std::unordered_set<MarkerOptions> &options);

visualization_msgs::Marker MarkerOfPoints(
    const std::vector<Eigen::Vector3d> &points, bool red);

constexpr MarkerOptions kRed = MarkerOptions::kRed;
constexpr MarkerOptions kGreen = MarkerOptions::kGreen;
constexpr MarkerOptions kCell = MarkerOptions::kCell;
constexpr MarkerOptions kCov = MarkerOptions::kCov;
constexpr MarkerOptions kNormal = MarkerOptions::kNormal;
