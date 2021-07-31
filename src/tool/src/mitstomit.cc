#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <common/common.h>

namespace po = boost::program_options;
using namespace std;

int main(int argc, char **argv) {
  string infile;
  int idx;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("infile", po::value<string>(&infile)->required(), "Path of the mits file")
      ("index", po::value<int>(&idx)->required(), "Matching id");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  vector<common::MatchInternal> mits;
  common::SerializeIn(infile, mits);
  common::SerializeOut("mit.ser", mits[idx]);
}