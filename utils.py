import numpy as np
import math
from cil.optimisation.functions import LeastSquares, L2NormSquared
from cil.optimisation.algorithms import FISTA
from cil.plugins.ccpi_regularisation.functions import FGP_TV

# from utilities import *
from cil.framework import  AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.processors import Slicer

"""
A helper function to compute the Herman-Meyer ordering for the subsets
Written with the help of Imraj Singh and Riccardo Barbano
"""
def herman_meyer_order(n):
    order = [0] * n
    factors = []
    len_order = len(order)

    while n % 2 == 0:
        factors.append(2)
        n //= 2

    # Check for odd factors
    for factor in range(3, int(n**0.5) + 1, 2):
        while n % factor == 0:
            factors.append(factor)
            n //= factor

    # If n is a prime number greater than 2
    if n > 2:
        factors.append(n)

    n_factors = len(factors)
    value = 0
    for factor_n in range(n_factors):
        n_change_value = 1 if factor_n == 0 else math.prod(factors[:factor_n])
        n_rep_value = 0

        for element in range(len_order):
            mapping = value
            n_rep_value += 1
            if n_rep_value >= n_change_value:
                value += 1
                n_rep_value = 0
            if value == factors[factor_n]:
                value = 0
            order[element] += math.prod(factors[factor_n+1:]) * mapping
    return order


def create_partition(data, num_angles, num_subsets, ig, device='cpu',partition_type='staggered'):

    if partition_type == 'staggered':
        datas = [Slicer(roi = {'angle' : (i, num_angles, num_subsets)})(data) for i in range(num_subsets)]
        Ais = [ProjectionOperator(ig, data_batch.geometry, device = device) for data_batch in datas]
        smooth_fs = [LeastSquares(Ai, b=datai, c=0.5) for Ai, datai in zip(Ais, datas)]
        smooth_preop_fs = [0.5*L2NormSquared(b=datai) for datai in datas]
    elif partition_type == 'sequential':
        datas = [Slicer(roi = {'angle' : (i, num_angles, num_subsets)})(data) for i in range(num_subsets)]
        Ais = [ProjectionOperator(ig, data_batch.geometry, device = device) for data_batch in datas]
        smooth_fs = [LeastSquares(Ai, b=datai, c=0.5) for Ai, datai in zip(Ais, datas)]
        smooth_preop_fs = [0.5*L2NormSquared(b=datai) for datai in datas]
    return Ais, datas, smooth_fs, smooth_preop_fs