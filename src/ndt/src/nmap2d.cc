#include <common/eigen_utils.h>
#include <ndt/nmap2d.h>

NMap2D::NMap2D(double cell_size)
    : cell_size_(cell_size), min_voxel_(Eigen::Vector2d::Zero()) {}

void NMap2D::LoadPoints(const std::vector<Eigen::Vector2d> &points) {
  std::vector<Eigen::Vector2d> valids;
  ExcludeInfinite(points, valids);
  ComputeVoxelOffset(valids);
  for (const auto &pt : valids) cells_[GetIndexForPoint(pt)].AddPoint(pt);
  for (auto &[idx, cell] : cells_) {
    Eigen::Vector2d c = idx.cast<double>() + Eigen::Vector2d(0.5, 0.5);
    cell.SetCenter(min_voxel_ + c * cell_size_);
    cell.SetSize(cell_size_);
  }
  std::vector<Eigen::Vector2d> means;
  for (auto &elem : cells_) {
    auto &cell = elem.second;
    cell.ComputeGaussian();
    if (cell.GetHasGaussian()) means.push_back(cell.GetMean());
  }
  MakeKDTree(means);
}

void NMap2D::LoadPointsWithCovariances(
    const std::vector<Eigen::Vector2d> &points,
    const Eigen::Matrix2d &point_cov) {
  std::vector<Eigen::Vector2d> valids;
  ExcludeInfinite(points, valids);
  ComputeVoxelOffset(valids);
  for (const auto &pt : valids)
    cells_[GetIndexForPoint(pt)].AddPointWithCovariance(pt, point_cov);
  for (auto &[idx, cell] : cells_) {
    Eigen::Vector2d c = idx.cast<double>() + Eigen::Vector2d(0.5, 0.5);
    cell.SetCenter(min_voxel_ + c * cell_size_);
    cell.SetSize(cell_size_);
  }
  std::vector<Eigen::Vector2d> means;
  for (auto &elem : cells_) {
    auto &cell = elem.second;
    cell.ComputeGaussian();
    if (cell.GetHasGaussian()) means.push_back(cell.GetMean());
  }
  MakeKDTree(means);
}

void NMap2D::LoadPointsWithCovariances(
    const std::vector<Eigen::Vector2d> &points,
    const std::vector<Eigen::Matrix2d> &point_covs) {
  if (points.size() != point_covs.size()) {
    std::cerr << __FUNCTION__ << ": unmatched sizes\n";
    std::exit(1);
  }
  std::vector<Eigen::Vector2d> vpts;
  std::vector<Eigen::Matrix2d> vcovs;
  ExcludeInfinite(points, point_covs, vpts, vcovs);
  ComputeVoxelOffset(vpts);
  for (size_t i = 0; i < vpts.size(); ++i)
    cells_[GetIndexForPoint(vpts[i])].AddPointWithCovariance(vpts[i], vcovs[i]);
  for (auto &[idx, cell] : cells_) {
    Eigen::Vector2d c = idx.cast<double>() + Eigen::Vector2d(0.5, 0.5);
    cell.SetCenter(min_voxel_ + c * cell_size_);
    cell.SetSize(cell_size_);
  }
  std::vector<Eigen::Vector2d> means;
  for (auto &elem : cells_) {
    auto &cell = elem.second;
    cell.ComputeGaussian();
    if (cell.GetHasGaussian()) means.push_back(cell.GetMean());
  }
  MakeKDTree(means);
}

std::vector<Cell2D> NMap2D::TransformCells(const Eigen::Affine2d &T,
                                           bool include_data) const {
  std::vector<Cell2D> ret;
  Eigen::Matrix2d R = T.rotation();
  Eigen::Vector2d t = T.translation();
  for (const auto &elem : cells_) {
    const auto &c = elem.second;
    Cell2D cell;
    cell.SetN(c.GetN());
    cell.SetHasGaussian(c.GetHasGaussian());
    cell.SetSize(c.GetSize());
    cell.SetCenter(T * c.GetCenter());
    cell.SetMean(R * c.GetMean() + t);
    if (c.GetHasGaussian()) {
      cell.SetCov(R * c.GetCov() * R.transpose());
      cell.SetEvals(c.GetEvals());
      cell.SetEvecs(R * c.GetEvecs());
      cell.SetNormal(R * c.GetNormal());
    }
    if (include_data) {
      cell.SetPoints(TransformPoints(c.GetPoints(), T));
      // Not set point_covs: current unnecessary
    }
    cell.SetCellType(c.GetCellType());
    cell.SetRescaleRatio(c.GetRescaleRatio());
    ret.push_back(cell);
  }
  return ret;
}

const Cell2D &NMap2D::SearchNearestCell(const Eigen::Vector2d &query) const {
  Eigen::Vector2i dummy;
  return SearchNearestCell(query, dummy);
}

const Cell2D &NMap2D::SearchNearestCell(const Eigen::Vector2d &query,
                                        Eigen::Vector2i &index) const {
  pcl::PointXYZ pt;
  pt.x = query(0), pt.y = query(1), pt.z = 0.;
  std::vector<int> idx{0};
  std::vector<float> dist2{0};
  kdtree_.nearestKSearch(pt, 1, idx, dist2);
  index = GetIndexForPoint(kdtree_pts_[idx[0]]);
  return cells_.at(index);
}

std::vector<std::reference_wrapper<const Cell2D>> NMap2D::SearchCellsInRadius(
    const Eigen::Vector2d &query, double radius) const {
  pcl::PointXYZ pt;
  pt.x = query(0), pt.y = query(1), pt.z = 0.;
  std::vector<int> idx;
  std::vector<float> dist2;
  kdtree_.radiusSearch(pt, radius, idx, dist2);
  std::vector<std::reference_wrapper<const Cell2D>> ret;
  for (auto id : idx) {
    Eigen::Vector2i index = GetIndexForPoint(kdtree_pts_[id]);
    ret.push_back(cells_.at(index));
  }
  return ret;
}

Eigen::Vector2i NMap2D::GetIndexForPoint(const Eigen::Vector2d &point) const {
  return ((point - min_voxel_) / cell_size_).array().floor().cast<int>();
}

void NMap2D::ComputeVoxelOffset(const std::vector<Eigen::Vector2d> &points) {
  Eigen::Vector2d min_pt;
  min_pt.fill(std::numeric_limits<double>::max());
  for (const auto &pt : points) {
    min_pt(0) = std::min(min_pt(0), pt(0));
    min_pt(1) = std::min(min_pt(1), pt(1));
  }
  min_voxel_ = (min_pt / cell_size_).array().floor() * cell_size_;
}

void NMap2D::MakeKDTree(const std::vector<Eigen::Vector2d> &points) {
  kdtree_pts_.clear();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto &pt : points) {
    pcl::PointXYZ p;
    p.x = pt(0), p.y = pt(1), p.z = 0.;
    pc->push_back(p);
    kdtree_pts_.push_back(pt);
  }
  kdtree_.setInputCloud(pc);
}
