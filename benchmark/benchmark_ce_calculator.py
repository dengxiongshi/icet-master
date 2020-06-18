import numpy as np
import time
from ase.build import bulk

from icet import ClusterSpace, ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator


def time_local_energy(calculator, occupations, iters):
    """ Returns timing for local energy calculations. """
    t0 = time.time()
    for _ in range(iters):
        calculator._calculate_local_contribution(occupations=occupations,
                                                 index=2)
    t1 = time.time() - t0
    return t1


def time_total_energy(calculator, occupations, iters):
    """ Returns timing for total energy calculations. """
    t0 = time.time()
    for _ in range(iters):
        calculator.calculate_total(occupations=occupations)
    t1 = time.time() - t0
    return t1


def print_timing_ratios(structure, local_iters, total_iters, sizes, cutoffs):
    """ Prints timing ratios between local and total energy calculations. """

    print('# 1:size 2:local_iters 3:total_iters 4:atom size, 5:ce calc init time (sec)'
          ', 6:t_local 7:t_total 8:t_total/t_local')
    cs = ClusterSpace(structure, cutoffs, chemical_symbols=['Al', 'Ga'])
    parameters = np.array([1.2 for _ in range(len(cs))])
    ce = ClusterExpansion(cs, parameters)
    for size in sizes:
        structure_cpy = structure.repeat(size)
        occupations = structure_cpy.get_atomic_numbers()
        t0 = time.time()
        calculator = ClusterExpansionCalculator(structure_cpy, ce)
        time_ce_init = time.time() - t0

        t_local = time_local_energy(calculator, occupations, local_iters)
        t_total = time_total_energy(calculator, occupations, total_iters)
        print(size, local_iters, total_iters, len(structure_cpy), time_ce_init,
              t_local, t_total, t_total / t_local)


if __name__ == '__main__':

    local_iters = 5000
    total_iters = 1
    structure = bulk('Al')
    cutoffs = [10, 6]
    chemical_symbols = ['Al', 'Ga']
    sizes = [4, 6, 8, 10, 14, 16, 20]
    print_timing_ratios(structure, local_iters, total_iters, sizes, cutoffs)
