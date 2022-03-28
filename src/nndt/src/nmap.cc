#include <common/eigen_utils.h>
#include <nndt/nmap.h>

NMap::NMap(double cell_size)
    : cell_size_(cell_size), min_voxel_(Eigen::Vector3d::Zero()) {}

void NMap::LoadPoints(const std::vector<Eigen::Vector3d> &points) {
  std::vector<Eigen::Vector3d> valids;
  ExcludeInfinite(points, valids);
  ComputeVoxelOffset(valids);
  for (const auto &pt : valids) cells_[GetIndexForPoint(pt)].AddPoint(pt);
  for (auto &[idx, cell] : cells_) {
    Eigen::Vector3d c = idx.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5);
    cell.SetCenter(min_voxel_ + c * cell_size_);
  }
  std::vector<Eigen::Vector3d> means;
  for (auto &elem : cells_) {
    elem.second.ComputeGaussian();
    if (elem.second.GetHasGaussian()) means.push_back(elem.second.GetMean());
  }
  MakeKDTree(means);
}

void NMap::LoadPointsWithCovariances(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<Eigen::Matrix3d> &point_covs) {
  if (points.size() != point_covs.size()) {
    std::cerr << __FUNCTION__ << ": unmatched sizes\n";
    std::exit(1);
  }
  std::vector<Eigen::Vector3d> vpts;
  std::vector<Eigen::Matrix3d> vcovs;
  ExcludeInfinite(points, point_covs, vpts, vcovs);
  ComputeVoxelOffset(vpts);
  for (size_t i = 0; i < vpts.size(); ++i)
    cells_[GetIndexForPoint(vpts[i])].AddPointWithCovariance(vpts[i], vcovs[i]);
  for (auto &[idx, cell] : cells_) {
    Eigen::Vector3d c = idx.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5);
    cell.SetCenter(min_voxel_ + c * cell_size_);
  }
  std::vector<Eigen::Vector3d> means;
  for (auto &elem : cells_) {
    elem.second.ComputeGaussian();
    if (elem.second.GetHasGaussian()) means.push_back(elem.second.GetMean());
  }
  MakeKDTree(means);
}

std::vector<Cell> NMap::TransformCells(const Eigen::Affine3d &T,
                                       bool include_data) const {
  std::vector<Cell> ret;
  Eigen::Matrix3d R = T.rotation();
  Eigen::Vector3d t = T.translation();
  for (const auto &elem : cells_) {
    const Cell &c = elem.second;
    Cell cell;
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
    cell.SetTolerance(c.GetTolerance());
    ret.push_back(cell);
  }
  return ret;
}

const Cell &NMap::SearchNearestCell(const Eigen::Vector3d &query) const {
  Eigen::Vector3i dummy;
  return SearchNearestCell(query, dummy);
}

const Cell &NMap::SearchNearestCell(const Eigen::Vector3d &query,
                                    Eigen::Vector3i &index) const {
  pcl::PointXYZ pt;
  pt.x = query(0), pt.y = query(1), pt.z = query(2);
  std::vector<int> idx{0};
  std::vector<float> dist2{0};
  int found = kdtree_.nearestKSearch(pt, 1, idx, dist2);
  if (!found) {
    std::cerr << __FUNCTION__ << ": Search Failed\n";
    std::exit(1);
  }
  pcl::PointXYZ res = kdtree_.getInputCloud()->at(idx[0]);
  Eigen::Vector3d mean(res.x, res.y, res.z);
  index = GetIndexForPoint(mean);
  return cells_.at(index);
}

Eigen::Vector3i NMap::GetIndexForPoint(const Eigen::Vector3d &point) const {
  return ((point - min_voxel_) / cell_size_).array().floor().cast<int>();
}

void NMap::ComputeVoxelOffset(const std::vector<Eigen::Vector3d> &points) {
  Eigen::Vector3d min_pt;
  min_pt.fill(std::numeric_limits<double>::max());
  for (const auto &pt : points) {
    min_pt(0) = std::min(min_pt(0), pt(0));
    min_pt(1) = std::min(min_pt(1), pt(1));
    min_pt(2) = std::min(min_pt(2), pt(2));
  }
  min_voxel_ = (min_pt / cell_size_).array().floor() * cell_size_;
}

void NMap::MakeKDTree(const std::vector<Eigen::Vector3d> &points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto &pt : points) {
    pcl::PointXYZ p;
    p.x = pt(0), p.y = pt(1), p.z = pt(2);
    pc->push_back(p);
  }
  kdtree_.setInputCloud(pc);
}
