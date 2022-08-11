#pragma once

enum class Options {
  kNDT,
  kNVNDT,
  kSICP,
  k1to1,
  k1ton,
  kIterative,
  kPointCov,
  kNoReject,
  k2D,
  k3D,
  kAnalytic,
  kUseNormalOMP,
  kOneTime,
};

constexpr Options kNDT = Options::kNDT;
constexpr Options kNVNDT = Options::kNVNDT;
constexpr Options kSICP = Options::kSICP;
constexpr Options k1to1 = Options::k1to1;
constexpr Options k1ton = Options::k1ton;
constexpr Options kIterative = Options::kIterative;
constexpr Options kPointCov = Options::kPointCov;
constexpr Options kNoReject = Options::kNoReject;
constexpr Options k2D = Options::k2D;
constexpr Options k3D = Options::k3D;
constexpr Options kAnalytic = Options::kAnalytic;
constexpr Options kUseNormalOMP = Options::kUseNormalOMP;
constexpr Options kOneTime = Options::kOneTime;
