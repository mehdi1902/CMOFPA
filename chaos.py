# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 11:37:04 2015

@author: 8
"""

from math import exp
import matplotlib.pyplot as plt


class Chaos_map():
    def __init__(self, map_type='logistic'):
        self.map_type = map_type
            
    def gauss_map(self, n, x1, alpha, beta):
        X = [x1]
        for i in range(n-1):
            X.append(exp(-alpha * X[-1]**2) + beta)        
        return X
    
    def logistic_map(self, n, x1, r):
        X = [x1]
        for i in range(n-1):
            X.append(r * X[-1] * (1-X[-1]))
        return X

    def tent_map(self, n, x1, mu):
        X = [x1]
        for i in range(n-1):
            X.append(X[-1]<.5 and mu*X[-1] or mu*(1-X[-1]))
        return X
    
if __name__=='__main__':
    #X = gauss_map(40, 2, .5, 1)
    c = Chaos_map()
    Y = c.logistic_map(200, .01, 3.6)
#    Y = c.tent_map(20, .6, 3)
    X = range(len(Y))
    
    plt.hist(Y[:10])
#    plt.plot(X, Y)
    plt.show()



