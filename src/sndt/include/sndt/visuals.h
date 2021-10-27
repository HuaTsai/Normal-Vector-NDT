/**
 * @file visuals.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Declaration of Visual Utilities
 * @version 0.1
 * @date 2021-07-30
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <sndt/ndt_map.h>
#include <sndt/sndt_map.h>
#include <visualization_msgs/MarkerArray.h>

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

enum class Color {
  kRed,    /**< (R, G, B) = (1.0, 0.0, 0.0) */
  kLime,   /**< (R, G, B) = (0.0, 1.0, 0.0) */
  kBlue,   /**< (R, G, B) = (0.0, 0.0, 1.0) */
  kWhite,  /**< (R, G, B, A) = (x, x, x, 0.0) */
  kBlack,  /**< (R, G, B) = (0.0, 0.0, 0.0) */
  kGray,   /**< (R, G, B) = (0.5, 0.5, 0.5) */
  kYellow, /**< (R, G, B) = (1.0, 1.0, 0.0) */
  kAqua,   /**< (R, G, B) = (0.0, 1.0, 1.0) */
  kFuchsia /**< (R, G, B) = (1.0, 0.0, 1.0) */
};

/**
 * @brief Factory method for making @c std_msgs::ColorRGBA
 *
 * @param color Input color defined in enum @c Color
 * @param alpha Alpha, 0 represents transparent while 1 represents opaque
 * @see Color
 */
std_msgs::ColorRGBA MakeColorRGBA(const Color &color, double alpha = 1.);

/**
 * @brief Given an ellipse and an external point, find the tangent points
 *
 * @param ellipse Input ellipse
 * @param point Input external point
 * @return Two tangent points
 */
std::vector<Eigen::Vector2d> FindTangentPoints(const Marker &ellipse,
                                               const Eigen::Vector2d &point);

/**
 * @brief Join markers by assigning different id
 *
 * @param markers Input markers
 * @return Joined markerarray
 */
MarkerArray JoinMarkers(const std::vector<Marker> &markers);

/**
 * @brief Join markerarrays by assigning different id
 *
 * @param markerarrays Input markerarrays
 * @return Joined markerarray
 */
MarkerArray JoinMarkerArrays(const std::vector<MarkerArray> &markerarrays);

/**
 * @brief Join markerarrays and markers by assigning different id
 *
 * @param markerarrays Input markerarrays
 * @param markers Input markers
 * @return Joined markerarray
 */
MarkerArray JoinMarkerArraysAndMarkers(
    const std::vector<MarkerArray> &markerarrays,
    const std::vector<Marker> &markers);

/**
 * @brief Marker of the cell boundary
 *
 * @param center Cell center
 * @param size Cell size
 * @param skew_rad Cell skew radian (default 0)
 * @param color Boundary Color (default kBlack)
 * @param alpha Alpha (default 1.0)
 * @return Marker of type @c LINE_LIST of the cell boundary
 */
Marker MarkerOfBoundary(const Eigen::Vector2d &center,
                        double size,
                        double skew_rad = 0,
                        const Color &color = Color::kBlack,
                        double alpha = 1.0);

/**
 * @brief Marker of the cell covariance
 *
 * @param mean Ellipse center
 * @param covariance Ellipse axes and length
 * @param color Ellipse color (default kLime)
 * @param alpha Alpha (default 0.6)
 * @return Marker of type @c SPHERE the cell covariance
 */
Marker MarkerOfEllipse(const Eigen::Vector2d &mean,
                       const Eigen::Matrix2d &covariance,
                       const Color &color = Color::kLime,
                       double alpha = 0.6);

/**
 * @brief Marker of the circle
 *
 * @param center Circle center
 * @param size Circle size
 * @param color Circle color (default kLime)
 * @param alpha Alpha (default 0.6)
 * @return Marker of type @c SPHERE of the circle
 */
Marker MarkerOfCircle(const Eigen::Vector2d &center,
                      double size,
                      const Color &color = Color::kLime,
                      double alpha = 0.6);

/**
 * @brief Marker of the lines
 *
 * @param points Points that form lines
 * @param color Lines color
 * @param alpha Alpha (default 1.0)
 * @return Marker of type @c LINE_LIST of the lines
 * @pre The size of @c points should be even
 * @details Points 0-1 forms line 0, points 2-3 forms line 1, etc.
 */
Marker MarkerOfLines(const std::vector<Eigen::Vector2d> &points,
                     const Color &color = Color::kLime,
                     double alpha = 1.0);

/**
 * @brief Marker of the lines by middle points
 *
 * @param points Points that form lines
 * @param color Lines color (default kLime)
 * @param alpha Alpha (default 1.0)
 * @return Marker of type @c LINE_LIST of the lines
 * @details Points 0-1 forms line 0, points 1-2 forms line 1, etc.
 */
Marker MarkerOfLinesByMiddlePoints(const std::vector<Eigen::Vector2d> &points,
                                   const Color &color = Color::kLime,
                                   double alpha = 1.0);

/**
 * @brief Marker of the points
 *
 * @param points Input points
 * @param size Input point size
 * @param color Points color (default kLime)
 * @param alpha Alpha (default 1.0)
 * @return Marker of type @c SPHERE_LIST of the points
 */
Marker MarkerOfPoints(const std::vector<Eigen::Vector2d> &points,
                      double size = 0.5,
                      const Color &color = Color::kLime,
                      double alpha = 1.0);

/**
 * @brief Marker of the arrows
 *
 * @param start Input arrow start
 * @param end Input arrow end
 * @param color Arrow color (default kLime)
 * @param alpha Alpha (default 1.0)
 * @return Marker of type @c ARROW of the arrow
 */
Marker MarkerOfArrow(const Eigen::Vector2d &start,
                     const Eigen::Vector2d &end,
                     const Color &color = Color::kLime,
                     double alpha = 1.0);

/**
 * @brief Marker of the text
 *
 * @param text Input text
 * @param position Text position
 * @param color Text color (default kBlack)
 * @param alpha Alpha (default 1.0)
 * @return Marker of type @c TEXT_VIEW_FACING of the text
 */
Marker MarkerOfText(std::string text,
                    const Eigen::Vector2d &position,
                    const Color &color = Color::kBlack,
                    double alpha = 1.0);

/**
 * @brief MarkerArray of the arrows
 *
 * @param starts Starts of the arrows
 * @param ends Ends of the arrows
 * @param color Arrows color (default kLime)
 * @param alpha Alpha (default 1.0)
 * @pre The size of @c starts and @c ends should be the same
 * @return MarkerArray of arrows
 */
MarkerArray MarkerArrayOfArrows(const std::vector<Eigen::Vector2d> &starts,
                                const std::vector<Eigen::Vector2d> &ends,
                                const Color &color = Color::kLime,
                                double alpha = 1.0);

/**
 * @brief General MarkerArray of SNDTCell
 *
 * @param cell Input cell pointer
 * @return General MarkerArray of SNDTCell
 * @details Boundary, point covariance, (normal covariance, and tangent lines)
 * are drawed.
 */
MarkerArray MarkerArrayOfSNDTCell(const SNDTCell *cell);

/**
 * @brief Special MarkerArray of SNDTCell
 *
 * @param cell Input cell pointer
 * @return Special MarkerArray of SNDTCell
 * @note Boundary, point covariance, (normal covariance, and tangent lines)
 * are drawed. The only difference is that the covariance color and the boundary
 * color are red.
 */
MarkerArray MarkerArrayOfSNDTCell2(const SNDTCell *cell);

/**
 * @brief MarkerArray of SNDTMap
 *
 * @param map Input SNDTMap
 * @param use_target_color Use target color (default false)
 * @return MarkerArray of SNDTMap
 * @note This function omits those HasGaussian() is false
 */
MarkerArray MarkerArrayOfSNDTMap(const SNDTMap &map,
                                 bool use_target_color = false);

/**
 * @brief MarkerArray of SNDTMap
 *
 * @param map Input SNDTMap
 * @param use_target_color Use target color (default false)
 * @return MarkerArray of SNDTMap
 * @note This function omits those HasGaussian() is false
 */
MarkerArray MarkerArrayOfSNDTMap(
    const std::vector<std::shared_ptr<SNDTCell>> &map,
    bool use_target_color = false);

/**
 * @brief MarkerArray of correspondences
 *
 * @param source_cell Source cell
 * @param target_cell Target cell
 * @param text Text of the cost value
 * @param color Text color (default black)
 * @return MarkerArray of correspondences
 */
MarkerArray MarkerArrayOfCorrespondences(const NDTCell *source_cell,
                                         const NDTCell *target_cell,
                                         std::string text,
                                         const Color &color = Color::kBlack);

MarkerArray MarkerArrayOfCorrespondences(
    const SNDTMap &smap,
    const SNDTMap &tmap,
    const Eigen::Affine2d &aff,
    const std::vector<std::pair<int, Eigen::Vector2d>> &corres);

/**
 * @brief MarkerArray of the sensors
 *
 * @param affs Poses of the sensors
 * @return MarkerArray of the sensors
 */
MarkerArray MarkerArrayOfSensor(const std::vector<Eigen::Affine2d> &affs);
