import numpy as np, pandas as pd
from scipy import stats
rng=np.random.default_rng(12345)
def pear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if (a.std()>0 and b.std()>0 and len(a)>2) else np.nan
def pearp(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.std()==0 or b.std()==0 or len(a)<3: return np.nan,np.nan
    r,p=stats.pearsonr(a,b); return float(r),float(p)
def fisher_mean(rs):
    rs=np.clip(np.array([r for r in rs if np.isfinite(r)],float),-0.999,0.999)
    return float(np.tanh(np.mean(np.arctanh(rs)))) if len(rs) else np.nan
def vboot_r(x,y,B=10000):
    x=np.asarray(x,float);y=np.asarray(y,float);n=len(x)
    idx=rng.integers(0,n,size=(B,n))
    xs=x[idx];ys=y[idx]
    xm=xs.mean(1,keepdims=True);ym=ys.mean(1,keepdims=True)
    xc=xs-xm;yc=ys-ym
    num=(xc*yc).sum(1); den=np.sqrt((xc**2).sum(1)*(yc**2).sum(1))
    r=np.divide(num,den,out=np.full(B,np.nan),where=den>0)
    r=r[np.isfinite(r)]; return np.percentile(r,[2.5,97.5])
