from netCDF4 import Dataset
import numpy as np
from datetime import datetime
import os,glob
import genutils as gu

# top level function for getting np array from wrfout
# searches cache for pre extracted raw vars, pre calced azim av, then creates if neccesary
# force=True to force recalc, z=0 for surface, z='full' for all, z=[n1, n2,...] for model levels
def getWRF(wopath,varin,vtype='raw',force=False,z=0):

    assert wopath[:4]=='/net',"wopath is absolute path"
    rname=os.path.basename(os.path.dirname(wopath))
    woname=os.path.basename(wopath)
# def getWRF(fname,varin,vtype='raw',force=False,z=0,nlay=100,intlog=True):

    pydatadir='/net/wrfstore6/disk1/nsparks/itc/interm/'
#     wrfdatadir='/net/wrfstore6/disk1/nsparks/itc/run/'
    # add surface suffix to var
    if z==0:
        if varin in ['vt','vr','V','th','v','u']:
            var=varin+'10'
        if varin in ['P']:
            var=varin+'sfc'
        if varin in ['T']:
            var=varin+'2'
        if varin in ['cc']:
            var=varin
    else:
        var=varin
    
    # check if var,run,z,vtype exists. In not, create
    gwname=woname + '.' + var + '.' + vtype + '.npy' #getWRF name
    gwpath=os.path.join(pydatadir,rname,gwname)
    
    
    if (not os.path.exists(gwpath)) or force==True :
        print('Creating: ' + os.path.join(rname,gwname))
        sdir=os.path.join(pydatadir,rname)
        if not os.path.exists(sdir):
            os.makedirs(sdir)
#         wrfoutpath=os.path.join(wrfdatadir,rname,woname)
        spath=os.path.join(sdir,woname)

        if vtype=='raw':
            if z==0 and var in ['vt10', 'vr10', 'V10', 'th10']:
                lon0,lat0=getWRF(wopath,'cc',force=force)
                V10, vt10, vr10, th10=wrf2pol(wopath,'sfc',lon0,lat0)
                np.save(spath + '.' + 'V10' + '.' + vtype, V10)
                np.save(spath + '.' + 'vt10' + '.' + vtype, vt10)
                np.save(spath + '.' + 'vr10' + '.' + vtype, vr10)
                np.save(spath + '.' + 'th10' + '.' + vtype, th10)
            elif z!=0 and var in ['vt', 'vr', 'V', 'th']:
                lon0,lat0=getWRF(wopath,'cc',force=force)
                V, vt, vr, th=wrf2pol(wopath,'full',lon0,lat0)
                np.save(spath + '.' + 'V' + '.' + vtype, V)
                np.save(spath + '.' + 'vt' + '.' + vtype, vt)
                np.save(spath + '.' + 'vr' + '.' + vtype, vr)
                np.save(spath + '.' + 'th' + '.' + vtype, th)
            
            elif z==0 and var=='u10':
                x=wrfout2var2d(wopath,'U10')
                np.save(spath+ '.' + var + '.' +vtype, x)
            elif z==0 and var=='v10':
                x=wrfout2var2d(wopath,'V10')
                np.save(spath+ '.' + var + '.' +vtype, x)
                        
            elif z!=0 and var=='u':
                x=wrfout2var3d(wopath,'U')
                np.save(spath+ '.' + var + '.' +vtype, x)
            elif z!=0 and var=='v':
                x=wrfout2var3d(wopath,'V')
                np.save(spath+ '.' + var + '.' +vtype, x)
                
                
            elif z==0 and var=='T2':
                T2=wrfout2var2d(wopath,'T2')
                np.save(spath + '.' + 'T2' + '.' + vtype,T2)
            elif z!=0 and var=='TH':
                TH=wrfout2var3d(wopath,'T')+300  
                np.save(spath + '.' + 'TH' + '.' + vtype,TH)
            elif z!=0 and var=='T':
                TH=getWRF(wopath,'TH',force=force,z='full',vtype='raw')
                P=getWRF(wopath,'P',force=force,z='full',vtype='raw')
                T=TH2T(TH,P)
                np.save(spath + '.' + 'T' + '.' + vtype,T)   
            elif z==0 and var=='Psfc':
                Psfc=wrf2P(wopath,'sfc')
                np.save(spath + '.' + 'Psfc' + '.' + vtype ,Psfc)
            elif z!=0 and var=='P':
                P=wrf2P(wopath,'full')
                np.save(spath + '.' + 'P' + '.' + vtype,P)
            elif z!=0 and var=='H':
                H=wrfout2var3d(wopath,'PH')
                HB=wrfout2var3d(wopath,'PHB')
                np.save(spath + '.' + 'H' + '.' + vtype,(H+HB)/9.81)
            elif z==0 and var=='cc': # cyclone centre indices
                P=getWRF(wopath,'P',z=10) #~950hPa
                xmin,ymin=np.unravel_index(P.argmin(),P.shape)
                np.save(spath + '.' + var + '.' + vtype,(xmin,ymin))
            else:
                print('Bad Raw Variable: ' + var)
        
        elif vtype=='az':
            # print('az')
            if z==0:
                zaz=0
            else:
                zaz='full'
            x=getWRF(wopath,varin,vtype='raw',force=force,z=zaz)
            lon0,lat0=getWRF(wopath,'cc',force=force)
            xaz,r=azimAv(x,lon0,lat0)
            np.save(spath + '.' + var + '.' + vtype, xaz)
            
        elif vtype=='sm':
            x=getWRF(wopath,varin,vtype='raw',force=force,z=z)
#             print(type(x))
            xsm=gu.smooth2d(x,6)
#             print(type(xsm))
            np.save(spath + '.' + var + '.' + vtype, xsm)
            
        elif vtype=='dwcm': # density weighted column mean
            assert(z=='full')
            H=getWRF(wopath,'H',force=force,z='full',vtype='raw')
            P=getWRF(wopath,'P',force=force,z='full',vtype='raw')
            T=getWRF(wopath,'T',force=force,z='full',vtype='raw')
            zd=np.diff(H,axis=2)/9.8
            wgt=zd*P/T
            x=getWRF(wopath,varin,vtype='raw',force=force,z='full')
            wgtx=wgt*x
            xdwcm=np.sum(wgtx,axis=2)/np.sum(wgt,axis=2)
            np.save(spath+ '.' + var + '.' + vtype,xdwcm)
             
        #pressure level height weighted column mean !!!
        #this is the correct method for mass weighted column means
        elif vtype=='dpwcm': 
            assert(z=='full')
            P=getWRF(wopath,'P',force=force,z='full',vtype='raw')
            x=getWRF(wopath,varin,vtype='raw',force=force,z='full')
            xc=np.sum(0.5*(x[:,:,:-1]+x[:,:,1:])*np.diff(P,axis=2),axis=2)/np.sum(np.diff(P,axis=2),axis=2)
            np.save(spath+ '.' + var + '.' + vtype,xc)   
         #pressure level height weighted column mean of smoothed var   
        elif vtype=='dpwsmcm': 
            assert(z=='full')
            P=getWRF(wopath,'P',force=force,z='full',vtype='raw')
            x=getWRF(wopath,varin,vtype='sm',force=force,z='full')
            xc=np.sum(0.5*(x[:,:,:-1]+x[:,:,1:])*np.diff(P,axis=2),axis=2)/np.sum(np.diff(P,axis=2),axis=2)
            np.save(spath+ '.' + var + '.' + vtype,xc) 
            
        elif vtype=='azdwcm':
            assert(z=='full')
            lon0,lat0=getWRF(wopath,'cc',force=force)
            xdwcm=getWRF(wopath,varin,vtype='dwcm',force=force,z='full')
            xaz,r=azimAv(xdwcm,lon0,lat0)
            np.save(spath+ '.' + var + '.'+ vtype,xaz)
        
        elif vtype=='azdpwcm':
            assert(z=='full')
            lon0,lat0=getWRF(wopath,'cc',force=force)
            xdpwcm=getWRF(wopath,varin,vtype='dpwcm',force=force,z='full')
            xaz,r=azimAv(xdpwcm,lon0,lat0)
            np.save(spath+ '.' + var + '.'+ vtype,xaz)
            
        elif vtype=='azdpwsmcm':
            assert(z=='full')
            lon0,lat0=getWRF(wopath,'cc',force=force)
            xdpwcm=getWRF(wopath,varin,vtype='dpwsmcm',force=force,z='full')
            xaz,r=azimAv(xdpwcm,lon0,lat0)
            np.save(spath+ '.' + var + '.'+ vtype,xaz)
            
            
        else:
            print('Bad vtype: ' + vtype)
    # Get Var (should have been created above, or existed already)
    #print('Loading: ' + os.path.join(rname,fname) + '.' + var + '.' + vtype + '.npy')
    x=np.load(gwpath)
    if z in [0,'full']:
        pass
    else:
        if vtype=='raw':
            x=x[:,:,z-1]
        elif vtype=='az':
            x=x[:,z-1]
            # pass # z index not required as performed on 1st call to raw
    return(x)

# # creates evenly spaced pressure grid for interpgrid2np func
# def equalPgrid(P,n):
#     Pi=np.full((P.shape[0],P.shape[1],n),np.nan)
#     for i in range(P.shape[0]):
#         for j in range(P.shape[1]):
#             Pi[i,j,:]=np.array(range(0,n))*(P[i,j,0]-P[i,j,-1])/(n-1)+P[i,j,-1]
#     return(Pi)

# # interps x at P to Pi
# def interpgrid2nP(x,P,Pi,intlog=True):
#     assert(x.shape==P.shape)
#     xi=np.full((x.shape[0],x.shape[1],Pi.shape[2]),np.nan)
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             if intlog:
#                 xci=-np.interp(-np.log(Pi[i,j,:]),-np.log(P[i,j,:]),-x[i,j,:])
#             else:
#                 xci=-np.interp(-(Pi[i,j,:]),-(P[i,j,:]),-x[i,j,:])
#             xi[i,j,:]=xci         
#     return(xi)


def wopath(run,fname):
    wopath=os.path.join('/net/wrfstore6/disk1/nsparks/itc/run',run,fname)
    return(wopath)
              
def getflist(run='run_CTRL',d=3):
    plist=sorted(glob.glob(os.path.join('/net/wrfstore6/disk1/nsparks/itc/run',run,'wrfout_d0' + str(d)+'*00')))
    flist=[os.path.basename(p) for p in plist]
    return(flist)

def wrf2pol(f,z,lon0,lat0):
    if z=='sfc':
        u=wrfout2var2d(f,'U10')
        v=wrfout2var2d(f,'V10')
    elif z=='full':
        u=wrfout2var3d(f,'U')
        v=wrfout2var3d(f,'V')
    else:
        print('Bad z:',z)
    vt,vr=cart2pol(u,v,lon0,lat0)
    #vtaz,r=azimAv(vt)
    #vraz,r=azimAv(vr)
    th=(np.arctan2(vt,vr)-np.pi/2)*180/np.pi
    V=np.sqrt(np.power(u,2) + np.power(v,2))
    #Vaz,r=azimAv(V)
    return(V,vt,vr,th)

def wrf2P(f,z):
    if z=='sfc':
        p=wrfout2var2d(f,'PSFC')
        pb=0
    elif z=='full':
        p=wrfout2var3d(f,'P')
        pb=wrfout2var3d(f,'PB')
    else:
        print('Bad z:',z)
    return((p+pb)/100)

def TH2T(TH,P):
    T=TH*(P/1000)**0.2854
    return(T)

def wrf2max(run,f,var,minim=False):
    t=getElapsedDays(run,f)
    x,y=getCoords(run,f)
    dx=np.diff(x)[0]
    v=getWRF(run,f,var,vtype='az')
    vmax,rmax,zmax=getvmax(v)
    rmax=rmax*dx/1000
    if minim:
        vmax,rmax,zmax=getvmax(-v)
        vmax=-vmax
    return(vmax,rmax,zmax,t)

def wrf2r(run,f,var,rs,minim=False):
    t=getElapsedDays(wopath(run,f))
    x,y=getCoords(wopath(run,f))
    dx=np.diff(x)[0]
    v=getWRF(run,f,var,vtype='az')
    rc=getRcoord(wopath(run,f))
    vr=[]
    for r in rs:
        assert(r>=np.min(rc) and r<= np.max(rc))
        rx=np.argmin( np.abs(r-rc))
        vr.append(v[rx])
    vr.append(t)
    return(vr)

def getElapsedDays(run,f):
    fpath=wopath(run,f)
    ncd=Dataset(fpath)
    x=ncd.variables['Times'][:].data[0]
    x=[i.decode('UTF-8') for i in x]
    outtime=''.join(x)
    starttime=ncd.SIMULATION_START_DATE
    dt0=datetime.strptime(starttime, "%Y-%m-%d_%H:%M:%S")
    dt1=datetime.strptime(outtime, "%Y-%m-%d_%H:%M:%S")
    dt=dt1-dt0
    days=dt.days+dt.seconds/(24*60*60)
    return(days)

def getvmax(xrz):
    if xrz.ndim == 1: # reshape to 2d
        #x=np.reshape(x,(1,x.shape[0],x.shape[1]))
        xrz=np.expand_dims(xrz,1)
    xm=np.nanmax(xrz)
    xmx=np.where(xrz == xm)
    rmax=xmx[0][0]
    zmax=xmx[1][0]
    return(xm,rmax,zmax)

def getCoords(runname,woname,corners=False,cc=True,force=False):  
    fpath=wopath(runname,woname)
    nc = Dataset(fpath,'r')
    nx=nc.dimensions['west_east'].size
    ny=nc.dimensions['south_north'].size
    dx=nc.DX
    dy=nc.DY
    nc.close()
    assert(dx==dy)
    dxy=dx
    x=np.linspace(-(nx-1)/2,(nx-1)/2,nx)*dxy
    y=np.linspace(-(ny-1)/2,(ny-1)/2,ny)*dxy
    if corners: #useful for pcolor(mesh)
        x=np.linspace(-(nx)/2,(nx)/2,nx+1)*dxy
        y=np.linspace(-(ny)/2,(ny)/2,ny+1)*dxy
    if cc: # cyclone centred coords
        lonx0,latx0=getWRF(runname,woname,'cc',force=force)
        lon0=x[lonx0] 
        lat0=y[latx0]
        x=x+lon0
        y=y+lat0
    return(x,y)

def getHeightCoord(run,fname):
    woname=wopath(run,fname)
    nc = Dataset(woname,'r')
    phb=nc.variables['PHB']
    z=np.mean(phb[0,:,:,:],axis=(1,2))/9.8
    nc.close()
    return(z.data)

def getRcoord(f):
    nc = Dataset(f,'r')
    nx=nc.dimensions['west_east'].size
    ny=nc.dimensions['south_north'].size
    dx=nc.DX
    nc.close()
    rbins=np.arange(1,nx/2,1)
    r=rbins*dx/1000
    return(r)

def wrfout2var2d(ncfname,var):
# returns wrfout 2d var as np array (lon,lat,z), ignores tx > 0
    nc = Dataset(ncfname,'r')
    v=nc.variables[var][0,:,:].filled(np.nan)
    nc.close()
    return(v)

def wrfout2var3d(ncfname,var):
# returns wrfout 3d var as np array (lon,lat,z), ignores tx > 0
    ncd = Dataset(ncfname,'r')
    v = ncd.variables[var][0,:,:,:].filled(np.nan)
    ncd.close()
    v = unstagger(v)
    v= np.transpose(v,(-2,-1,0))
    return(v)

# def unstagger_old(u):
#     nd=u.ndim
#     if nd==2: tr=(0,1)
#     if nd==3: tr=(1,2,0)
#     if nd==4: tr=(2,3,0,1)
#     u=np.transpose(u,tr)
#     if u.shape[0]==u.shape[1]:
#         0 # do nothing
#     elif u.shape[0]==u.shape[1]+1:
#             u=u[:-1,:]
#     elif u.shape[0]+1 == u.shape[1]:
#             u=u[:,:-1]
#     else:
#         print('Strange dimensions:',u.shape)
#     if nd==2: tr=(0,1)
#     if nd==3: tr=(2,0,1)
#     if nd==4: tr=(2,3,0,1)
#     u=np.transpose(u,tr)
#     return(u)

def unstagger(u):
    nd=u.ndim
    if nd==2: tr=(0,1)
    if nd==3: tr=(1,2,0)
    if nd==4: tr=(2,3,0,1)
    u=np.transpose(u,tr)
    if u.shape[0]==u.shape[1]:
        0 # do nothing
    elif u.shape[0]==u.shape[1]+1:
            u=0.5*(u[:-1,:]+u[1:,:])
    elif u.shape[0]+1 == u.shape[1]:
            u=0.5*(u[:,:-1]+u[:,1:])
    else:
        print('Strange dimensions:',u.shape)
    if nd==2: tr=(0,1)
    if nd==3: tr=(2,0,1)
    if nd==4: tr=(2,3,0,1)
    u=np.transpose(u,tr)
    return(u)

def cart2pol(u,v,x0=-1,y0=-1):
# assumes dims 0,1 are x,y coords
    dims=u.ndim
    if dims == 2: # add trailing singleton
        u=np.expand_dims(u,2)
        v=np.expand_dims(v,2)
    nx=u.shape[0]
    ny=v.shape[1]
    if x0==-1:
        x0=nx/2
    if y0==-1:
        y0=ny/2
    x=np.arange(0,nx)-x0
    y=np.arange(0,ny)-y0
    xg,yg=np.meshgrid(y,x)
    dg=np.sqrt(np.power(xg,2)+np.power(yg,2))
    dg[np.where(dg==0)]=1
    cosg=np.divide(xg,dg)
    sing=np.divide(yg,dg)
    cosg=np.expand_dims(cosg,2)
    sing=np.expand_dims(sing,2)
    ut=v*np.tile(cosg,[1,1,v.shape[2]]) - \
        u*np.tile(sing,[1,1,v.shape[2]])
    ur=v*np.tile(sing,[1,1,v.shape[2]]) + \
        u*np.tile(cosg,[1,1,v.shape[2]])
    return(np.squeeze(ut),np.squeeze(ur))

def rplane(shp,d,x0,y0):
    nx=shp[0]
    ny=shp[1]
    if x0==-1:
        x0=nx/2
    if y0==-1:
        y0=ny/2
    y=(np.tile(range(ny),[nx,1])-y0)*d
    x=(np.tile(np.arange(nx).reshape((nx,1)),[1,ny])-x0)*d
    r=np.sqrt(np.square(x)+np.square(y))
    return(r)

def azimAv(v,x0=-1,y0=-1):
# works on 2d and 3d. Expectsd x,y as first 2 dims
    dims=len(v.shape)
    if dims == 2: # reshape to 3d
        v=np.expand_dims(v,2)
    dxy=1
    r=rplane(v.shape,1,x0,y0)
    dr=dxy #rad bin width
    rbinmax=v.shape[0]/2
    rbins=np.arange(1,rbinmax,dr)
    rbm=[]
    for n in range(len(rbins)):
        rx=np.where((r>rbins[n]) & (r<rbins[n]+dr))
        #print(v.shape)
        #print(v[rx[0],rx[1],:].shape)
        #print((np.mean(v[rx[0],rx[1],:],axis=0)).shape)
        rbm.append(np.mean(v[rx[0],rx[1],:],axis=0))
    rbm=np.stack(rbm).astype(np.single)
    #rzout={'az':rbm, 'r':rbins, 'z':z[:-1]}
    return(np.squeeze(rbm),rbins)
