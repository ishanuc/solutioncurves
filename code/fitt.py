#!/usr/bin/python
# code for estimating solution curve for classification  problems
# USAGE: ./fitt.py [filename] [outfilename]
# [filename] has two columns x y , x is the data amount, y is mutual information
# produces multiple output files with [outfilename] as prefix
#------------------------------------
import os
import inspect

import numpy as np
import sys
from numpy import array
from scipy.optimize import leastsq
import matplotlib as mpl
import matplotlib.pyplot as plt

def __residual(params, y, x):
    A, B, P = params
    EP=0
    if P > 1:
        EP=10
    if B < 1:
        EP=10
    
    return y + A+ B/(x**P)  + EP

OUTFILE='D'
DATFILE='cpos'
total = len(sys.argv)
if total > 1:
    DATFILE = sys.argv[1]
if total > 2:
    OUTFILE = sys.argv[1]

x,y = np.loadtxt(DATFILE, skiprows=0, unpack=True,comments='%' )

x=np.array(x)
y=np.array(y)

#initial values
A=1
B=1
P=1

p_opt,infodict = leastsq(__residual,  array([A,B,P]), args=(y, x),maxfev=1000000)

print 'output params', p_opt
print 'message ( 1,2,3,4 => soln found) ', infodict

ERR_=__residual(p_opt, y, x)
print 'ERRS=',ERR_
print 'sum of lsq.= ', np.sum(np.abs(ERR_))

A, P, dd =p_opt 
plt.semilogx(x,2**(y-A),'ro')
xx=np.logspace(0,2,100)
plt.semilogx(xx,2**(-B/(xx**P)),'-k')

if OUTFILE!="D":
    y1=y/np.exp(dd)
    y2=np.exp(-A/(xx**P))
    plt.savefig(OUTFILE + '.png',dpi=600, bbox_inches='tight')
    np.savetxt(OUTFILE+'_res', np.c_[x,y1], delimiter=' ') 
    np.savetxt(OUTFILE+'_res_', np.c_[xx,y2], delimiter=' ') 
else:
    plt.show() 