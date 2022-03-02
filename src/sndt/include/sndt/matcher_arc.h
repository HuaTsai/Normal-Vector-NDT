
Eigen::Affine2d Pt2plICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    Pt2plICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

// BUG: This match does not work and it is awful.
Eigen::Affine2d P2DNDTMDMatch(
    const NDTMap &target_map,
    const std::vector<Eigen::Vector2d> &source_points,
    P2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d P2DNDTMatch(
    const NDTMap &target_map,
    const std::vector<Eigen::Vector2d> &source_points,
    P2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d D2DNDTMDMatch(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMDMatch(
    const SNDTMap &target_map,
    const SNDTMap &source_map,
    SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMatch(
    const SNDTMap &target_map,
    const SNDTMap &source_map,
    SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMDMatch2(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());
