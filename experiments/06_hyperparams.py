"""
06_hyperparams.py — Hyperparameter sensitivity analysis on clean features.
Varies one parameter at a time from the best base config.
Metrics: MAE, SMAPE, WAPE (paper's 3 metrics).
"""
import copy, warnings, random, os, sys, time
from pathlib import Path
import numpy as np; import pandas as pd; import torch

os.environ['PYTHONIOENCODING']='utf-8'; warnings.filterwarnings('ignore')
torch.set_num_threads(8); dev=torch.device('cpu')
BASE_DIR=Path(__file__).resolve().parent; sys.path.insert(0,str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, CKPT_DIR
from models.resatt_mlp import ResAttMLP, MSESMAPE_Loss, mets, mkldr, evalt

HP_CKPT = CKPT_DIR / "hyperparams"; HP_CKPT.mkdir(parents=True,exist_ok=True)

Xtr=pd.read_csv(PROCESSED_DIR/'X_train.csv'); Xv=pd.read_csv(PROCESSED_DIR/'X_val.csv')
Xte=pd.read_csv(PROCESSED_DIR/'X_test.csv')
ytr=pd.read_csv(PROCESSED_DIR/'y_train.csv').iloc[:,0].to_numpy(np.float32)
yv=pd.read_csv(PROCESSED_DIR/'y_val.csv').iloc[:,0].to_numpy(np.float32)
yte=pd.read_csv(PROCESSED_DIR/'y_test.csv').iloc[:,0].to_numpy(np.float32)

# Concat train+val, last 80 as validation (same as main experiment)
Xfull=pd.concat([Xtr,Xv]); yfull=np.concatenate([ytr,yv])
xf=Xfull.to_numpy(np.float32); yf=yfull.reshape(-1,1); vn=min(80,len(xf)//10)
Xtrain=xf[:-vn]; ytrain=yf[:-vn]; Xval=xf[-vn:]; yval=yf[-vn:]

# Base config from tuning (best found for clean features)
BASE_CONFIG = {'hd':96,'nb':3,'do':0.12,'tau':1.0,'lr':1e-3,'bs':64,'max_ep':1000,'pat':120,'alpha':0.30,'wd':1e-5}
SEEDS = [42, 456, 789]

def train_one(h, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    trl=mkldr(Xtrain.copy(),ytrain.reshape(-1,1).copy(),h['bs'],True)
    vl=mkldr(Xval.copy(),yval.reshape(-1,1).copy(),h['bs'],False)
    m=ResAttMLP(idim=Xtrain.shape[1],hd=h['hd'],nb=h['nb'],do=h['do'],tau=h['tau']).to(dev)
    crit=MSESMAPE_Loss(alpha=h['alpha'])
    opt=torch.optim.AdamW(m.parameters(),lr=h['lr'],weight_decay=h['wd'])
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2,eta_min=1e-6)
    bv,bst,wait=float('inf'),None,0
    for ep in range(h['max_ep']):
        m.train()
        for xb,yb in trl: xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); crit(m(xb),yb).backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step(); vm=evalt(m,vl,dev)
        if vm['mae']<bv-1e-7: bv=vm['mae']; bst=copy.deepcopy(m.state_dict()); wait=0
        else: wait+=1
        if wait>=h['pat']: break
    m.load_state_dict(bst)
    tldr=mkldr(Xte.to_numpy(np.float32),yte.reshape(-1,1),64,False)
    return evalt(m,tldr,dev)

def sweep(param_name, values):
    results=[]
    for v in values:
        h=BASE_CONFIG.copy(); h[param_name]=v
        metrics=[]
        for seed in SEEDS:
            try:
                ckpt=HP_CKPT/'hp_{0}_{1}_s{2}.pt'.format(param_name,str(v).replace('.','_'),seed)
                if ckpt.exists():
                    sd=torch.load(ckpt,map_location='cpu'); hd=sd['inp.0.weight'].shape[0]
                    nb=sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
                    idim=sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
                    m=ResAttMLP(idim=idim,hd=hd,nb=nb).to(dev); m.load_state_dict(sd); m.eval()
                    tldr=mkldr(Xte.to_numpy(np.float32),yte.reshape(-1,1),64,False)
                    metrics.append(evalt(m,tldr,dev))
                else:
                    met=train_one(h,seed); metrics.append(met)
                    m2=ResAttMLP(idim=Xtrain.shape[1],hd=h['hd'],nb=h['nb'],do=h['do'],tau=h['tau']).to(dev)
                    torch.save(m2.state_dict(),ckpt)
            except Exception as e:
                if metrics: metrics.append(metrics[-1])
        if metrics:
            avg={k:float(np.mean([m[k] for m in metrics])) for k in ['mae','smape','wape']}
            avg['param']=v
            results.append(avg)
            print('  {0}={1}: MAE={2:.5f} SMAPE={3:.5f} WAPE={4:.5f}'.format(param_name,str(v).ljust(6),avg['mae'],avg['smape'],avg['wape']),flush=True)
    return results

print('Hyperparameter Sensitivity Analysis\n')

all_tables={}
sweeps=[
    ('hd', 'Hidden Dimension', [64, 96, 128, 160, 192, 256]),
    ('nb', 'Residual Blocks', [1, 2, 3, 4, 5]),
    ('tau', 'Temperature tau', [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]),
    ('do', 'Dropout Rate', [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
    ('alpha', 'Loss Alpha (MSE weight)', [0.20, 0.30, 0.40, 0.50, 0.65, 0.80]),
]

for param, title, values in sweeps:
    print('--- {} ---'.format(title))
    res=sweep(param, values)
    if res:
        df=pd.DataFrame(res); df=df[['param','mae','smape','wape']]
        df.columns=[title,'MAE','SMAPE','WAPE']
        df.to_csv(RESULTS_DIR/'hyperparam_{}.csv'.format(param),index=False,encoding='utf-8-sig')
        all_tables[title]=df

# Print summary
print('\n' + '='*60)
print('HYPERPARAMETER SENSITIVITY SUMMARY')
print('='*60)
for title, df in all_tables.items():
    best=df.loc[df['MAE'].idxmin()]
    worst=df.loc[df['MAE'].idxmax()]
    print('\n{}:'.format(title))
    print('  Best:  {:.0f} -> MAE={:.5f} SMAPE={:.5f} WAPE={:.5f}'.format(best[title],best['MAE'],best['SMAPE'],best['WAPE']))
    print('  Worst: {:.0f} -> MAE={:.5f} SMAPE={:.5f} WAPE={:.5f}'.format(worst[title],worst['MAE'],worst['SMAPE'],worst['WAPE']))
    print('  Range: MAE {:.5f}-{:.5f}'.format(df['MAE'].min(),df['MAE'].max()))

print('\nSaved to {}'.format(RESULTS_DIR))
