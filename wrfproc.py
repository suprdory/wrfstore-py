# functions used by proc_runs in data processing workflow

import numpy as np
import wrftools as wrf
import pandas as pd

def get_Pbar(Pr,r): #exact dicrete mean P (within radius r)
    Par0=Pr[0]*r[0]**2
    Par=Pr[1:]*(r[1:]**2-r[:-1]**2)
    return(np.cumsum(np.insert(Par,0,Par0))/(r**2))
    
def get_rain_in_r(x,y,tp,r):
    xg,yg=np.meshgrid(x,y)
    rg=np.sqrt(xg**2+yg**2)
    return(np.sum(tp[rg<r]))

def proc_precip(df):
    tp500=[np.nan]
    tpmax=[np.nan]
    rtpmax=[np.nan]
    tpmaxpoint=[np.nan]
    dfd=df.diff()
    df['dtp']=dfd.tpgrid
    for row in df[1:].iterrows():
        r=row[1]
        tpaz=wrf.azimAv(r.dtp,r.x0,r.y0)[0]
        tpmax_r,rtpmax_r,ztpmax=wrf.getvmax(tpaz)
        rtpmax.append(rtpmax_r*(r.xc[1]-r.xc[0])/1000)
        tpmax.append(tpmax_r) 
        tp500.append(get_rain_in_r(r.xc,r.yc,r.dtp,500000))
        tpmaxpoint.append(np.max(r.dtp))    
    df['tp500']=tp500
    df['tpmaxpoint']=tpmaxpoint
    df['tpmax']=tpmax
    df['rtpmax']=rtpmax
    df=df.drop(columns=['tpgrid', 'dtp','x0','y0','xc','yc'])
    return df
    
def get_row(fname,run,force):
    fpath=wrf.wopath(run,fname)
    V10max,rV10max,zmax,t=wrf.wrf2max(fpath,'V',force=force)
    vt10max,rvt10max,zmax,t=wrf.wrf2max(fpath,'vt',force=force)
    Pmin,rPmin,Pzmin,t=wrf.wrf2max(fpath,'P',minim=True,force=force)
    trst=wrf.getElapsedDays(fpath,rst=True)
    tlf=np.round(t-trst,5)
    Pr=wrf.getWRF(fpath,'P','az',force=force)
    Vt=wrf.getWRF(fpath,'vt','az',force=force)
    r=wrf.getRcoord(fpath)
    tpgrid=wrf.getWRF(fpath,'tp',force=force)
    xc,yc=wrf.getCoords(fpath,cc=True)
    x0,y0=wrf.getWRF(fpath,'cc')

    dr=r[1]-r[0]
    Pbar=get_Pbar(Pr,r)
    row = {
        'tlf':tlf,'t':t,'vt10max':vt10max,'rvt10max':rvt10max,\
        'V10max':V10max,'rV10max':rV10max,\
        'Pmin':Pmin,'Pr':Pr,'Pbar':Pbar,'r':r,'Vt10':Vt,\
        'tpgrid':tpgrid,'xc':xc,'yc':yc,'x0':x0,'y0':y0
    }
    return(row)
    
def get_run_max_df(flist,run,force=False):
    rows=[]
    for fname in flist:
        print(fname)
        rows.append(get_row(fname,run,force))
    df=pd.DataFrame(rows)
    return(df)


def create_delta_df(df):
    def antidiff_df(df): 
    # calc rolling 2 row mean, builds upon pd rolling for numerical list elements 
        dfm=df.rolling(2).mean()
        missingkeys=(np.setdiff1d(df.keys(),dfm.keys())) # missing because rolling ignores arrays
#         print(missingkeys)
        for k in missingkeys:
#             print(k)
            dfk=df[k]#.reset_index(drop=True)
#             print(dfk)
            dfkm=[np.nan]
            for i in range(len(dfk)-1):
#                 print(i)
                ar=np.array(np.mean((dfk[i],dfk[i+1]),0))
                dfkm.append(ar)
            dfm[k]=dfkm
        return(dfm)

    dfd=df.diff()
    dfm=antidiff_df(df)

    dfm['dPbardt']=dfd['Pbar']/(60*60)
    Vr=[]
    dPrdr=[]
#     print(dfm)
    dr=dfm['r'][1][1]-dfm['r'][1][0]
    for i in range(len(dfm)):
        Vr.append((100*dfm['dPbardt'][i])*(dfm['r'][i]*1000)/(2*dfm['Pr'][i]*100))
        dPrdr.append(100*np.gradient(dfm['Pr'][i])/(dr*1000))
    dfm['Vrcol']=Vr
    dfm['dPrdr']=dPrdr
    dfm['tp500']=dfd['tp500']
    return(dfm)

def tabulate_df_ens(dflist):
    dfn = pd.DataFrame(columns=['tlf','t','V10max','rV10max','Pmin','P','r','Vt10','Vrcol','dPdr','tp500'])
    for df in dflist:
        for i,row in df[1:].iterrows():
        #     print(row['r'])
            tlf=row['tlf']
            t=row['t']
            rs=row['r']
            V10max=row['V10max']
            rV10max=row['rV10max']
            Pmin=row['Pmin']
            tp500=row['tp500']
        #     print('i=' + str(i))
            for j,r in enumerate(rs[rs<100]):
        #         print(j)
                newrow = pd.Series({
                    'tlf':tlf,'t':t,'V10max':V10max,'rV10max':rV10max,'Pmin':Pmin,'tp500':tp500,\
                    'P':row['Pr'][j],'r':r,'Vt10':row['Vt10'][j],'Vrcol':row['Vrcol'][j],'dPdr':row['dPrdr'][j]})
                dfn=dfn.append(newrow,ignore_index=True)
    return dfn