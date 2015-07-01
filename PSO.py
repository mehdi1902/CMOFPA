# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:51:01 2015

@author: 8
"""

from pyswarm import pso
import numpy as np
from math import pi, exp, sqrt
#
def ackley(chrom):
    chrom = np.array(chrom)
    sum1 = np.sum(chrom**2)
    sum2 = np.sum(np.cos(2*pi*chrom))
    n = len(chrom)
    f = -20 * exp(-.2 * sqrt(sum1/n)) - exp(sum2/n) + 20 + exp(1)
    return f

dim = 4
x, f = pso(ackley, [-30]*dim, [30]*dim, maxiter=10000,
    minfunc=1e-15, minstep=1e-16)

print f