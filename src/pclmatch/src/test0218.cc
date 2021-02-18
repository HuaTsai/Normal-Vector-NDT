#include <bits/stdc++.h>
#include "ICP.h"
#include "common/common.h"

using namespace std;
using namespace trimesh;

#define BUNNYPATH "/home/ee904/Desktop/HuaTsai/NormalNDT/Data/bunny/data/bun000.ply"

vector<double> TransAndAngle(xform xf) {
  double angle;
  dvec3 axis;
  decompose_rot(xf, angle, axis);
  dvec3 trans(xf[12], xf[13], xf[14]);
  return {degrees(angle), len(trans)};
}

int main(int argc, char **argv) {
  TriMesh *mesh1, *mesh2;    
  mesh1 = TriMesh::read(BUNNYPATH);
  mesh2 = TriMesh::read(BUNNYPATH);

  KDtree *kd1 = new KDtree(mesh1->vertices);
	KDtree *kd2 = new KDtree(mesh2->vertices);

  auto aff = common::Affine3dFromXYZRPY({1, 1, 1, 0.2, 0.3, 0.2});
  xform xf1 = xform(aff.matrix().data());
  xform xf2;
  vector<float> weights1, weights2;
  float err = ICP(mesh1, mesh2, xf1, xf2, kd1, kd2, weights1, weights2, 0.0f, 2, ICP_RIGID);
  auto diff = inv(xf1) * xf2;
  cout << TransAndAngle(diff).at(0) << ", " << TransAndAngle(diff).at(1) << endl;
}