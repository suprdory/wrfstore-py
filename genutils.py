import numpy as np
import time
import math


def avg_list(x,dt,t=0):
    if t==0:
        t=np.arange(0,len(x))
    xa=[]
    ta=[]
    nx=math.floor(len(x)/dt)
    for i in range(0,nx):
        ta.append(t[i*dt])
        xi=x[i*dt:(i+1)*dt]
        xis=np.stack(xi)
        xa.append(np.mean(xis,axis=0))
    return(xa,ta)


def smooth(x, N):
  return np.convolve(x, np.ones((N))/N)[(N-1):]

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

# TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(TicToc,tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic(TicToc):
#     TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
    # Records a time in TicToc, marks the beginning of a time interval
    toc(TicToc,tempBool=False)