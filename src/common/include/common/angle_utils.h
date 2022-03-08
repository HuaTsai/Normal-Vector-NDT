#pragma once
#include <cmath>

inline double Rad2Deg(double radian) {
  return radian * 180. / M_PI;
}

inline double Deg2Rad(double degree) {
  return degree * M_PI / 180.;
}
