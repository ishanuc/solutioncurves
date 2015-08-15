#!/usr/bin/python
# code for estimating solution curve for point estimation problems
# USAGE: ./fitt_h.py [filename] [outfilename]
# [filename] has two columns x y , x is the data amount, y is negative entropy
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
    W, A, B, P = params
    EP=0
    if P > 1:
        EP=10
    if B < 1:
        EP=10
    return y/W + A+ B/(x**P)  + EP

OUTFILE='D'
DATFILE='cpos'
total = len(sys.argv)
#initial values
W=50
A=-2
B=10
P=.4


if total > 1:
    DATFILE = sys.argv[1]
if total > 2:
    OUTFILE = sys.argv[2]
if total > 3:
    P = float(sys.argv[3])

x,y = np.loadtxt(DATFILE, skiprows=0, unpack=True,comments='%' )

x=np.array(x)
y=np.array(y)


p_opt,infodict = leastsq(__residual,  array([W,A,B,P]), args=(y, x),maxfev=1000000)

print 'output params', p_opt
print 'message ( 1,2,3,4 => soln found) ', infodict

ERR_=__residual(p_opt, y, x)
print 'ERRS=',ERR_
print 'sum of lsq.= ', np.sum(np.abs(ERR_))

W, A, B, P =p_opt 

plt.semilogx(x,2**(y/W+A),'ro')
xx=np.logspace(-10,12,100)
plt.semilogx(xx,2**(-( B/(xx**P))),'-k')


if OUTFILE!="D":
    y1=2**(y/W+A)
    y2=2**(-( B/(xx**P)))
    plt.savefig(OUTFILE + '.png',dpi=600, bbox_inches='tight')
    np.savetxt(OUTFILE+'_res', np.c_[x,y1], delimiter=' ') 
    np.savetxt(OUTFILE+'_res_', np.c_[xx,y2], delimiter=' ') 
else:
    plt.show() 
