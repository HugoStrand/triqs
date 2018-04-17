/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2017, H. U.R. Strand
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <triqs/clef.hpp>
#include <triqs/gfs.hpp>
#include <triqs/test_tools/gfs.hpp>

using namespace triqs::gfs;
using namespace triqs::arrays;
using namespace triqs::lattice;
using namespace triqs::clef;

namespace {

placeholder<0> iw;
placeholder<1> inu;
placeholder<2> k;
placeholder<3> r;

placeholder<4> a;
placeholder<5> b;
placeholder<6> c;
placeholder<7> d;

placeholder<8> inup;
placeholder<9> tau;

} // namespace

// ----------------------------------------------------

typedef gf<imfreq, matrix_valued> g_iw_t;
typedef g_iw_t::const_view_type g_iw_cvt;
typedef g_iw_t::view_type g_iw_vt;

typedef gf<brillouin_zone, matrix_valued> ek_t;
typedef ek_t::const_view_type ek_cvt;
typedef ek_t::view_type ek_vt;
  
typedef gf<cartesian_product<imfreq, brillouin_zone>, matrix_valued> gk_iw_t;
typedef gk_iw_t::const_view_type gk_iw_cvt;
typedef gk_iw_t::view_type gk_iw_vt;

typedef gf<cartesian_product<imfreq, cyclic_lattice>, matrix_valued> gr_iw_t;
typedef gr_iw_t::const_view_type gr_iw_cvt;
typedef gr_iw_t::view_type gr_iw_vt;

// ----------------------------------------------------

gk_iw_t g0k_from_ek(double mu, ek_vt ek, g_iw_t::mesh_t mesh) {

  gk_iw_t g0k = make_gf<gk_iw_t::mesh_t::var_t>({mesh, ek.mesh()}, ek.target());

  //  for (auto const &k : ek.mesh()) {

#pragma omp parallel for 
    for (int idx = 0; idx < ek.mesh().size(); idx++) {
      auto iter = ek.mesh().begin(); iter += idx; auto k = *iter;
    
    for (auto const &w : mesh)
      g0k[w, k](a, b) << kronecker(a, b) * (w + mu) - ek(k)(a, b);

    auto _ = var_t{};
    g0k[_, k] = inverse(g0k[_, k]);
  }

  return g0k;
}

gk_iw_t gk_from_ek_sigma(double mu, ek_vt ek, g_iw_vt sigma) {

  gk_iw_t gk =
      make_gf<gk_iw_t::mesh_t::var_t>({sigma.mesh(), ek.mesh()}, ek.target());

  //  gk(inu, k)(a, b) << kronecker(a, b) * (inu + mu) - ek(k)(a, b) -
  //                       sigma(inu)(a, b);

  //  for (auto const &k : ek.mesh()) {

#pragma omp parallel for 
  for (int idx = 0; idx < ek.mesh().size(); idx++) {
    auto iter = ek.mesh().begin(); iter += idx; auto k = *iter;
    
    for (auto const &w : sigma.mesh())
      gk[w, k](a, b) << kronecker(a, b) * (w + mu) - ek(k)(a, b) - sigma[w](a, b);

    auto _ = var_t{};
    gk[_, k] = inverse(gk[_, k]);
  }

  // gk = inverse(gk);  // does not work, see TRIQS issue #463
  return gk;
}

// ----------------------------------------------------

gr_iw_t gr_from_gk(gk_iw_vt gk) {

  auto wmesh = std::get<0>(gk.mesh());
  auto kmesh = std::get<1>(gk.mesh());
  auto lmesh = gf_mesh<cyclic_lattice>{kmesh.domain().lattice(), kmesh.periodization_matrix};

  gr_iw_t gr = make_gf<gr_iw_t::mesh_t::var_t>({wmesh, lmesh}, gk.target());

  for (auto const &w : wmesh) {
    auto _ = var_t{};
    gr[w, _] = inverse_fourier(gk[w, _]);
  }

  return gr;
}

gk_iw_t gk_from_gr(gr_iw_vt gr) {

  auto wmesh = std::get<0>(gr.mesh());
  auto lmesh = std::get<1>(gr.mesh());
  auto kmesh = gf_mesh<brillouin_zone>{brillouin_zone{lmesh.domain()}, lmesh.periodization_matrix};
  
  gk_iw_t gk = make_gf<gk_iw_t::mesh_t::var_t>({wmesh, kmesh}, gr.target());

  for (auto const &w : wmesh) {
    auto _ = var_t{};
    gk[w, _] = fourier(gr[w, _]);
  }

  return gk;
}

// ----------------------------------------------------

TEST(lattice, g0k_to_from_g0r) {
 double beta = 100.0;
 int n_iw = 1025;

 int nk = 4; 
 double t = 1.0;
 auto bz = brillouin_zone{bravais_lattice{{{1, 0}, {0, 1}}}};
 
 triqs::clef::placeholder<0> om_;
 triqs::clef::placeholder<1> k_;

 auto ek = ek_t{{bz, nk}, {1, 1}};
 ek(k_) << - 2*t * (cos(k_(0)) + cos(k_(1)));

 double mu = 0.;
 auto mesh = g_iw_t::mesh_t{beta, Fermion, n_iw};
 auto g0k = g0k_from_ek(mu, ek, mesh);

 std::cout << "--> gr_from_gk" << std::endl;
 auto g0r = gr_from_gk(g0k);
 std::cout << "--> gk_from_gr" << std::endl;
 auto g0k_ref = gk_from_gr(g0r);

 EXPECT_CLOSE_ARRAY(g0k.data(), g0k_ref.data()); 
}

TEST(lattice, gk_to_from_gr) {
 double beta = 100.0;
 int n_iw = 1025;

 int nk = 4; 
 double t = 1.0;
 auto bz = brillouin_zone{bravais_lattice{{{1, 0}, {0, 1}}}};
 
 triqs::clef::placeholder<0> om_;
 triqs::clef::placeholder<1> k_;

 auto ek = ek_t{{bz, nk}, {1, 1}};
 ek(k_) << - 2*t * (cos(k_(0)) + cos(k_(1)));

 double mu = 0.;
 auto mesh = g_iw_t::mesh_t{beta, Fermion, n_iw};

 auto sigma = g_iw_t{mesh, {1, 1}};
 sigma(om_) << 1./om_;
 
 auto gk = gk_from_ek_sigma(mu, ek, sigma);

 std::cout << "--> gr_from_gk" << std::endl;
 auto gr = gr_from_gk(gk);
 std::cout << "--> gk_from_gr" << std::endl;
 auto gk_ref = gk_from_gr(gr);

 EXPECT_CLOSE_ARRAY(gk.data(), gk_ref.data()); 
}

MAKE_MAIN;
