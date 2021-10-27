/**
 * @file contour.cc
 * @brief Output to /tmp/contour.txt, and Input of contour.py
 * @version 0.1
 * @date 2021-10-25
 * @copyright Copyright (c) 2021
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  Affine3d aff3 =
      Translation3d(0.943713, 0.000000, 1.840230) *
      Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 =
      Translation2d(aff3.translation()(0), aff3.translation()(1)) *
      Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, m;
  double cell_size, huber, voxel, x, y, t, r, radius;
  string data;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("huber,u", po::value<double>(&huber)->default_value(1.0), "Use Huber loss")
      ("radius,a", po::value<double>(&radius)->default_value(1.5), "Use Huber loss")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
      ("x,x", po::value<double>(&x)->default_value(0), "x")
      ("y,y", po::value<double>(&y)->default_value(0), "y")
      ("t,t", po::value<double>(&t)->default_value(0), "t")
      ("r,r", po::value<double>(&r)->default_value(15), "r")
      ("m,m", po::value<int>(&m)->default_value(1), "r");
}





