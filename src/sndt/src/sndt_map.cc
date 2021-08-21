/**
 * @file sndt_map.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Definition of SNDTMap
 * @version 0.1
 * @date 2021-07-29
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <sndt/sndt_map.h>

SNDTMap::SNDTMap(double cell_size) : MapInterface(cell_size) {}

SNDTMap::~SNDTMap() {
  if (is_loaded_) {
    for (auto &cell : active_cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
}

void SNDTMap::LoadPointsAndNormals(const std::vector<Eigen::Vector2d> &points,
                                   const std::vector<Eigen::Vector2d> &normals) {
  Expects(points.size() == normals.size());
  if (is_loaded_) {
    for (auto &cell : active_cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
  GuessMapSize(points);
  Initialize();
  std::vector<SNDTCell *> update_cells;
  for (size_t i = 0; i < points.size(); ++i) {
    Eigen::Vector2d point = points[i];
    Eigen::Vector2d normal = normals[i];
    SNDTCell *cell = GetCellAndAllocate(point);
    if (cell) {
      cell->AddPoint(point);
      cell->AddNormal(normal);
      update_cells.push_back(cell);
    }
  }
  for (auto cell : update_cells) cell->ComputeGaussian();
  is_loaded_ = true;
}

void SNDTMap::LoadPointsWithCovariancesAndNormals(
    const std::vector<Eigen::Vector2d> &points,
    const std::vector<Eigen::Matrix2d> &point_covs,
    const std::vector<Eigen::Vector2d> &normals) {
  Expects(points.size() == point_covs.size() && points.size() == normals.size());
  if (is_loaded_) {
    for (auto &cell : active_cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
  GuessMapSize(points);
  Initialize();
  std::vector<SNDTCell *> update_cells;
  for (size_t i = 0; i < points.size(); ++i) {
    Eigen::Vector2d point = points[i];
    Eigen::Vector2d normal = normals[i];
    Eigen::Matrix2d point_cov = point_covs[i];
    SNDTCell *cell = GetCellAndAllocate(point);
    if (cell) {
      cell->AddPointWithCovariance(point, point_cov);
      cell->AddNormal(normal);
      update_cells.push_back(cell);
    }
  }
  for (auto cell : update_cells) cell->ComputeGaussian();
  is_loaded_ = true;
}

std::vector<std::shared_ptr<SNDTCell>> SNDTMap::PseudoTransformCells(
    const Eigen::Affine2d &T, bool include_data) const {
  std::vector<std::shared_ptr<SNDTCell>> ret;
  Eigen::Matrix2d R = T.rotation();
  Eigen::Vector2d t = T.translation();
  double skew_rad = Eigen::Rotation2Dd(R).angle();
  for (auto it = begin(); it != end(); ++it) {
    auto cell = std::make_shared<SNDTCell>();
    cell->SetN((*it)->GetN());
    cell->SetPHasGaussian((*it)->GetPHasGaussian());
    cell->SetNHasGaussian((*it)->GetNHasGaussian());
    cell->SetSkewRad(skew_rad);
    cell->SetCenter(R * (*it)->GetCenter() + t);
    cell->SetSize((*it)->GetSize());
    cell->SetPointMean(R * (*it)->GetPointMean() + t);
    if ((*it)->GetPHasGaussian()) {
      cell->SetPointCov(R * (*it)->GetPointCov() * R.transpose());
      cell->SetPointEvals((*it)->GetPointEvals());
      cell->SetPointEvecs(R * (*it)->GetPointEvecs());
    }
    cell->SetNormalMean(R * (*it)->GetNormalMean());
    if ((*it)->GetNHasGaussian()) {
      cell->SetNormalCov(R * (*it)->GetNormalCov() * R.transpose());
      cell->SetNormalEvals((*it)->GetNormalEvals());
      cell->SetNormalEvecs(R * (*it)->GetNormalEvecs());
    }
    if (include_data) {
      for (auto pt : (*it)->GetPoints()) {
        if (pt.allFinite()) pt = R * pt + t;
        cell->AddPoint(pt);
      }
      for (auto nm : (*it)->GetNormals()) {
        if (nm.allFinite()) nm = R * nm;
        cell->AddNormal((nm(1) > 0) ? nm : -nm);
      }
    }
    ret.push_back(cell);
  }
  return ret;
}

SNDTCell *SNDTMap::GetCellAndAllocate(const Eigen::Vector2d &point) {
  if (!point.allFinite()) { return nullptr; }
  auto idx = GetIndexForPoint(point);
  if (!IsValidIndex(idx)) { return nullptr; }
  if (!cellptrs_[idx(0)][idx(1)]) {
    cellptrs_[idx(0)][idx(1)] = new SNDTCell();
    Eigen::Vector2d center;
    center(0) = map_center_(0) + (idx(0) - map_center_index_(0)) * cell_size_;
    center(1) = map_center_(1) + (idx(1) - map_center_index_(1)) * cell_size_;
    cellptrs_[idx(0)][idx(1)]->SetCenter(center);
    cellptrs_[idx(0)][idx(1)]->SetSize(cell_size_);
    active_cells_.push_back(cellptrs_[idx(0)][idx(1)]);
  }
  return cellptrs_[idx(0)][idx(1)];
}

const SNDTCell *SNDTMap::GetCellForPoint(const Eigen::Vector2d &point) const {
  if (!is_loaded_) { return nullptr; }
  auto idx = GetIndexForPoint(point);
  if (!IsValidIndex(idx)) { return nullptr; }
  return cellptrs_[idx(0)][idx(1)];
}

std::string SNDTMap::ToString() const {
  char c[600];
  sprintf(c, "cell size: %.2f\n"
             "map center: (%.2f, %.2f)\n"
             "map size: (%.2f, %.2f)\n"
             "cellptrs size: (%d, %d)\n"
             "center index: (%d, %d)\n"
             "active cells: %ld\n",
             cell_size_, map_center_(0), map_center_(1), map_size_(0),
             map_size_(1), cellptrs_size_(0), cellptrs_size_(1),
             map_center_index_(0), map_center_index_(1), active_cells_.size());
  std::string ret(c);
  for (size_t i = 0; i < active_cells_.size(); ++i) {
    auto idx = GetIndexForPoint(active_cells_[i]->GetCenter());
    char s[20];
    sprintf(s, "[#%ld, (%d, %d)] ", i, idx(0), idx(1));
    ret += std::string(s) + active_cells_[i]->ToString();
  }
  return ret;
}

std::vector<Eigen::Vector2d> SNDTMap::GetPoints() const {
  std::vector<Eigen::Vector2d> ret;
  for (auto it = begin(); it != end(); ++it)
    for (auto pt : (*it)->GetPoints())
      ret.push_back(pt);
  return ret;
}

std::vector<Eigen::Vector2d> SNDTMap::GetPointsWithGaussianCell() const {
  std::vector<Eigen::Vector2d> ret;
  for (auto it = begin(); it != end(); ++it)
    if ((*it)->HasGaussian())
      for (auto pt : (*it)->GetPoints())
        ret.push_back(pt);
  return ret;
}

std::vector<Eigen::Vector2d> SNDTMap::GetNormals() const {
  std::vector<Eigen::Vector2d> ret;
  for (auto it = begin(); it != end(); ++it)
    for (auto nm : (*it)->GetNormals())
      ret.push_back(nm);
  return ret;
}

void SNDTMap::Initialize() {
  cellptrs_ = new SNDTCell **[cellptrs_size_(0)];
  for (int i = 0; i < cellptrs_size_(0); ++i) {
    cellptrs_[i] = new SNDTCell *[cellptrs_size_(1)];
    memset(cellptrs_[i], 0, cellptrs_size_(1) * sizeof(SNDTCell *));
  }
}
