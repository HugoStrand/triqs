#include <triqs/test_tools/gfs.hpp>

using namespace triqs::gfs;
using namespace triqs::arrays;
using namespace triqs::lattice;

TEST(MeshBrillouinZone, OpenMP_for_loop) {

  int nk = 4;

  auto bz = brillouin_zone{bravais_lattice{{{1, 0}, {0, 1}}}};

  auto kmesh = gf_mesh<brillouin_zone>{bz, nk};

  for (auto const &k1 : kmesh) 
    for (auto const &k2 : kmesh) {
      std::cout << "k1, k2 = " << k1 << ", " << k2 << "; k1 + k2 = " << k1 + k2 << "\n";
      //std::cout << " k1 < k2 = " << (k1 < k2) << "\n";
    }

  for (auto const &k : kmesh) std::cout << k << "\n";

  std::cout << " ==== \n";

  //#pragma omp parallel for
  for (auto k_iter = kmesh.begin(); k_iter != kmesh.end(); k_iter++) {
    auto k = *k_iter;
    std::cout << k << "\n";
  }

  std::cout << " ==== \n";

  /*
#pragma omp parallel for
  for (auto k_iter = kmesh.begin(); k_iter <= kmesh.end(); k_iter++) {
    auto k = *k_iter;
    std::cout << k << "\n";
  }
  */
}
MAKE_MAIN;
