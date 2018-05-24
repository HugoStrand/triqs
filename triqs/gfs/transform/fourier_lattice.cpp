/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014 by O. Parcollet
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
/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by M. Ferrero, O. Parcollet
 * Copyright (C) 2018- by Simons Foundation
 *               authors : O. Parcollet, N. Wentzell, H. UR Strand
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
#include "../../gfs.hpp"
#include "./fourier_common.hpp"

namespace triqs::gfs {

  // The implementation is almost the same in both cases...
  template <typename V1, typename V2> void * __impl_plan(int fftw_backward_forward, gf_mesh<V1> const &out_mesh, gf_vec_cvt<V2> g_in) {

    //check periodization_matrix is diagonal
    for (int i = 0; i < g_in.mesh().periodization_matrix.shape()[0]; i++)
      for (int j = 0; j < g_in.mesh().periodization_matrix.shape()[1]; j++)
        if (i != j and g_in.mesh().periodization_matrix(i, j) != 0) TRIQS_RUNTIME_ERROR << "Periodization matrix must be diagonal for FFTW to work";

    auto g_out    = gf_vec_t<V1>{out_mesh, g_in.target_shape()[0]};
    int n_others = second_dim(g_in.data());

    auto dims = g_in.mesh().get_dimensions();
    auto p =_fourier_base_plan(g_in.data(), g_out.data(), dims.size(), dims.ptr(), n_others, fftw_backward_forward);

    return p;
  }

  void * _fourier_plan(gf_mesh<cyclic_lattice> const &r_mesh, gf_vec_cvt<brillouin_zone> gk) {
    return __impl_plan(FFTW_FORWARD, r_mesh, gk);
  }  

  void * _fourier_plan(gf_mesh<brillouin_zone> const &k_mesh, gf_vec_cvt<cyclic_lattice> gr) {
    return __impl_plan(FFTW_BACKWARD, k_mesh, gr);
  }

  
  // ------------------------ DIRECT TRANSFORM --------------------------------------------

  gf_vec_t<cyclic_lattice> _fourier_impl(gf_mesh<cyclic_lattice> const &r_mesh, gf_vec_cvt<brillouin_zone> gk, void * p) {
    auto gr = gf_vec_t<cyclic_lattice>{r_mesh, gk.target_shape()[0]};
    _fourier_base(gk.data(), gr.data(), p);
    gr.data() /= gk.mesh().size();
    return std::move(gr);
  }

  gf_vec_t<cyclic_lattice> _fourier_impl(gf_mesh<cyclic_lattice> const &r_mesh, gf_vec_cvt<brillouin_zone> gk) {
    auto p = _fourier_plan(r_mesh, gk);
    auto gr = _fourier_impl(r_mesh, gk, p);
    _fourier_destroy_plan(p);    
    return std::move(gr);
  }
  
  // ------------------------ INVERSE TRANSFORM --------------------------------------------

  gf_vec_t<brillouin_zone> _fourier_impl(gf_mesh<brillouin_zone> const &k_mesh, gf_vec_cvt<cyclic_lattice> gr, void * p) {
    auto gk = gf_vec_t<brillouin_zone>{k_mesh, gr.target_shape()[0]};
    _fourier_base(gr.data(), gk.data(), p);
    return std::move(gk);
  }

  gf_vec_t<brillouin_zone> _fourier_impl(gf_mesh<brillouin_zone> const &k_mesh, gf_vec_cvt<cyclic_lattice> gr) {
    auto p = _fourier_plan(k_mesh, gr);
    auto gk = _fourier_impl(k_mesh, gr, p);
    _fourier_destroy_plan(p);
    return std::move(gk);
  }
  
} // namespace triqs::gfs
