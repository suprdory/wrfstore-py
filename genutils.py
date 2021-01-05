import numpy as np
import time
import math
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


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
    return(np.convolve(x, np.ones((N))/N)[(N-1):])

def smooth2d(x,N):
    #smooth by layer
    if len(x.shape)==3:
        xsmz=[convolve2d(x[:,:,z], np.ones((N,N))/(N*N),mode='same') for z in range(0,x.shape[2])]
        xsm=np.stack(xsmz,axis=2)
    else:
        xsm=convolve2d(x, np.ones((N,N))/(N*N),mode='same')
    return(xsm)
    
def imshowxy(x,y,c,ax=0,**kwargs):
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    ext=(x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2)
    if ax==0:
        im=plt.imshow(c,origin='lower',extent=ext,**kwargs)
    else:
        im=ax.imshow(c,origin='lower',extent=ext,**kwargs)
    return(im)

def resample2center(v,cc):
#     cc=wrf.getWRF(run,fname,'cc')
    x0=int(v.shape[0]/2)-cc[0]
    y0=int(v.shape[1]/2)-cc[1]
    #print(v.shape,cc,(x0,y0))
    vcc=np.roll(v,(x0,y0),(0,1))
    #use np.roll but fill nans instead of roll
    
    return(vcc)

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