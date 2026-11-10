"""The supplied per-frame homographies (normalised pitch -> image pixels), and the camera each one implies."""
import numpy as np, pandas as pd
D=__import__('paths').DATA + ''
L,W=105.3,68.0
CX,CY=960.0,540.0

def load_h(src):
    h=pd.read_csv(f'{D}{src}/BAR-ATH_{src}_homography.csv')
    M=h[[f'h{r}{c}' for r in range(3) for c in range(3)]].values.reshape(-1,3,3)
    return M, h['valid'].values.astype(bool), h['reprojection_error'].values

def decompose(M):
    """M: (n,3,3) normalised-pitch -> pixel. Returns f, fb (two focal length estimates, px), R (n,3,3), t (n,3), C (n,3)
    camera centre in metres (pitch x, y; z negative = up: the pitch's y runs down, so its right-handed z points into
    the grass)."""
    S=np.diag([1/L,1/W,1.0])                     # metres -> normalised
    T=np.array([[1,0,-CX],[0,1,-CY],[0,0,1.0]])   # shift principal point to origin
    H=T@M@S                                       # metres -> centred pixels
    h1,h2,h3=H[:,:,0],H[:,:,1],H[:,:,2]
    # orthogonality of r1,r2 and equal norms -> focal length (two estimates)
    f2a=-(h1[:,0]*h2[:,0]+h1[:,1]*h2[:,1])/(h1[:,2]*h2[:,2])
    f2b=(h1[:,0]**2+h1[:,1]**2-h2[:,0]**2-h2[:,1]**2)/(h2[:,2]**2-h1[:,2]**2)
    f=np.sqrt(np.where(f2a>0,f2a,np.nan))
    fb=np.sqrt(np.where(f2b>0,f2b,np.nan))
    Kinv=np.zeros((len(H),3,3)); Kinv[:,0,0]=1/f; Kinv[:,1,1]=1/f; Kinv[:,2,2]=1
    r1=np.einsum('nij,nj->ni',Kinv,h1); r2=np.einsum('nij,nj->ni',Kinv,h2); tt=np.einsum('nij,nj->ni',Kinv,h3)
    lam=2/(np.linalg.norm(r1,axis=1)+np.linalg.norm(r2,axis=1))
    lam=np.where(tt[:,2]*lam<0,-lam,lam)         # pitch in front of the camera
    r1*=lam[:,None]; r2*=lam[:,None]; tt*=lam[:,None]
    r3=np.cross(r1,r2)
    R=np.stack([r1,r2,r3],axis=2)                 # columns r1 r2 r3
    good=np.all(np.isfinite(R),axis=(1,2))
    U,_,Vt=np.linalg.svd(R[good]); R[good]=U@Vt   # nearest rotation
    C=-np.einsum('nji,nj->ni',R,tt)               # C = -R^T t
    return f,fb,R,tt,C
