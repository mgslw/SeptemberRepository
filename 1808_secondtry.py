# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:44:30 2021

@author: magda
"""
import numpy as np
from numpy import linalg
rng = np.random.default_rng()
miki = []

ro = 1
eps = 0.0005
#G = rng.multivariate_normal([0,0,0,0], np.fill_diagonal(A, ro , 50)
'''fill diagonal doesn't work here, returns wrong shape'''
'''first we define function f, and functions for numerator and denominator'''
def f(x):
    return (np.linalg.norm(x))**2

def f1(x):
    return np.exp(-f(x))
cov_matrix = 0.5* np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
mean = np.array([1,0,2,0])


while np.linalg.norm(mean - 0) > eps: 
  '''second we draw a sample from 4d gaussian distribution, 50 instances'''
  sample = np.reshape(rng.multivariate_normal(mean, cov_matrix , 50), [4,50])

  '''calculate weights for each of 50 points and save them in a vector/array'''
  weight = np.apply_along_axis(f1, 0, sample)
  print(weight.shape)

  '''multiply each column by its weight'''
  for i in range (1,50):
      sample[:,i] *= weight[i]

  '''add up in each row'''
  numerator = np.sum(sample, 1)
  miki.append(numerator)

  '''add the weights'''
  denominator = np.sum(weight)
  print(numerator, denominator)

  '''divide and get x_i. THIS IS ACTUALLY THE NEW MEAN'''
  mean = numerator / denominator
 
norma_sredniej = np.apply_along_axis(np.linalg.norm, 1, miki)
 
import matplotlib.pyplot as plt
plt.plot(norma_sredniej)
plt.show()    
 
