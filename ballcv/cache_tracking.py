import sys, numpy as np, pandas as pd, re
D=__import__('paths').DATA + ''
OUT=__import__('paths').S
def load_atd(src):
    lines=open(f'{D}{src}/BAR-ATH_{src}_atd.txt').read().split('\n')
    lines=[l for l in lines if l]
    n=len(lines); ns=len(lines[0].split(':')[1].split(';'))
    P=np.full((n,ns,3),np.nan,np.float32); B=np.full((n,3),np.nan,np.float32)
    for i,l in enumerate(lines):
        fid,pl,ball=l.split(':')
        assert int(fid)==i+1
        if pl.count('NaN')<ns*3:
            P[i]=np.array(pl.replace(';',',').split(','),dtype=np.float32).reshape(ns,3)
        B[i]=np.array(ball.split(','),dtype=np.float32)
    xml=open(f'{D}{src}/BAR-ATH_{src}_atd.xml').read()
    teams=re.findall(r'<Team\b[^>]*\bid="([^"]+)"',xml)
    st=np.array([0 if t==teams[0] else 1 for t in re.findall(r'<Player\b[^>]*\bteamId="([^"]+)"',xml)])
    H=pd.read_csv(f'{D}{src}/BAR-ATH_{src}_homography.csv')
    return P,B,st,H['valid'].values.astype(bool),H['reprojection_error'].values
for src in ['tactical','broadcast']:
    P,B,st,valid,err=load_atd(src)
    np.savez(OUT+src+'.npz',P=P,B=B,team=st,valid=valid,err=err)
    nplay=np.isfinite(P[:,:,0]).sum(1)
    print(src,P.shape,'team sizes',np.bincount(st),'valid %.3f'%valid.mean(),'ball %.3f'%np.isfinite(B[:,0]).mean(),'mean players %.1f'%nplay.mean(), 'players when valid %.1f'%nplay[valid].mean())
