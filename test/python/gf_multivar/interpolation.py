# Copyright (c) 2025 Hugo U. R. Strand
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Hugo U. R. Strand


""" This is an attempt to show missing interpolation functionality in meshproduct Green's functions.

Currently a table of the interpolation/slice cases raising NotImplementedError is printed
and the test raises NotImplementedError if not all cases passes.
"""


import numpy as np

from itertools import product

from triqs.gf import *
from triqs.lattice import *


def type_name(m):
    return str(type(m)).split('.')[-1].split("'")[0]


def attempt_operation(code, label, mesh, target_shape):

    g = Gf(mesh=mesh, target_shape=target_shape)
    
    try:
        eval(code)
    except NotImplementedError:
        error = 'NotImplementedError'
    else:
        error = False

    result = (' x '.join([type_name(m) for m in mesh.components]), len(target_shape), label, error)
        
    return result


beta = 2.0
norb = 2

nk = 2
nr = 2

nw = 8
ntau = 10

dlr_opts = dict(w_max=1.0, eps=1e-8)

BL = BravaisLattice(units = [(1,0,0) , (0,1,0)])
BZ = BrillouinZone(BL)

rmesh = MeshCycLat(lattice=BL, dims=[nr, nr, nr])
kmesh = MeshBrZone(BZ, n_k=nk)

k0 = (0., 0., 0.)
r0 = (0, 0, 0)

spatial_meshes = [(r0, rmesh), (k0, kmesh)]

fwmesh = MeshImFreq(beta=beta, S='Fermion', n_iw=nw)
ftmesh = MeshImTime(beta=beta, S='Fermion', n_tau=ntau)

fDmesh = MeshDLR(beta=beta, statistic='Fermion', **dlr_opts)
fDwmesh = MeshDLRImFreq(beta=beta, statistic='Fermion', **dlr_opts)
fDtmesh = MeshDLRImTime(beta=beta, statistic='Fermion', **dlr_opts)

dyn_meshes = [fwmesh, ftmesh, fDmesh, fDwmesh, fDtmesh]

results = []


target_ranks = [2, 4]

# -- Slice the dynamic mesh using the 'all' operator

for target_rank, (x0, ms), md in product(target_ranks, spatial_meshes, dyn_meshes):
    results.append(attempt_operation(
        'g(x0, all)', 'Dynamic slice', MeshProduct(ms, md), [norb]*target_rank))

for target_rank, md, (x0, ms) in product(target_ranks, dyn_meshes, spatial_meshes):
    results.append(attempt_operation(
        'g(all, x0)', 'Dynamic slice', MeshProduct(md, ms), [norb]*target_rank))

# -- Slicing dynamic meshes: cases with multiple dynamic meshes 
    
for target_rank, (x0, ms), md1, md2 in product(target_ranks, spatial_meshes, dyn_meshes, dyn_meshes):
    results.append(attempt_operation(
        'g(x0, all, all)', 'Dynamic slice', MeshProduct(ms, md1, md2), [norb]*target_rank))

for target_rank, (x0, ms), md1, md2, md3 in product(target_ranks, spatial_meshes,
                                                    dyn_meshes, dyn_meshes, dyn_meshes):
    results.append(attempt_operation(
        'g(x0, all, all, all)', 'Dynamic slice', MeshProduct(ms, md1, md2, md3), [norb]*target_rank))
    

# -- Print results as a table (if pandas is available)

try:
    import pandas as pd

except ModuleNotFoundError:
    pass

else:
    pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)

    df = pd.DataFrame(results, columns=['Mesh 1', 'Target rank', 'Operation', 'Error'])
    print(df)

    
# -- Check if any exceptions were raised

errors = np.array([r[-1] for r in results])

if not (errors == 'False').all():
    raise NotImplementedError
