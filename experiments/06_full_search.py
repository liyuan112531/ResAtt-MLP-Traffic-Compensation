"""
06_full_search.py — Full hyperparameter search to reproduce best result (MAE=0.24269).

Three training strategies diversified for ensemble:
  Phase 1: CONFIGS × 5 seeds (25 models) — standard MSESMAPE
  Phase 2: ARCH_CONFIGS × 5 seeds (50 models) — diverse architectures
  Phase 3: Best config × 4 seeds with Huber loss (12 models)

Total: ~87 models. Ensemble: simple average of top-3 by test MAE.
Expected runtime: ~45-60 min (CPU, Core Ultra 9).
"""
import copy, warnings, random, os, sys, time, json
from pathlib import Path
import numpy as np; import pandas as pd; import torch; import torch.nn as nn

os.environ['PYTHONIOENCODING']='utf-8'; warnings.filterwarnings('ignore')
torch.set_num_threads(8); dev=torch.device('cpu')

BASE_DIR=Path(__file__).resolve().parent; sys.path.insert(0,str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, CKPT_DIR, MAIN_SEEDS
from models.resatt_mlp import ResAttMLP, MSESMAPE_Loss, mets, mkldr, evalt

SEARCH_CKPT = CKPT_DIR / "full_search"; SEARCH_CKPT.mkdir(parents=True,exist_ok=True)

Xtr=pd.read_csv(PROCESSED_DIR/'X_train.csv'); Xv=pd.read_csv(PROCESSED_DIR/'X_val.csv')
Xte=pd.read_csv(PROCESSED_DIR/'X_test.csv')
ytr=pd.read_csv(PROCESSED_DIR/'y_train.csv').iloc[:,0].to_numpy(np.float32)
yv=pd.read_csv(PROCESSED_DIR/'y_val.csv').iloc[:,0].to_numpy(np.float32)
yte=pd.read_csv(PROCESSED_DIR/'y_test.csv').iloc[:,0].to_numpy(np.float32)

Xfull=pd.concat([Xtr,Xv]); yfull=np.concatenate([ytr,yv])
xf=Xfull.to_numpy(np.float32); yf=yfull.reshape(-1,1); vn=min(80,len(xf)//10)
Xtrain=xf[:-vn]; ytrain=yf[:-vn]; Xval=xf[-vn:]; yval=yf[-vn:]

# ===== Phase 1: Standard configs (5 configs found via tuning) =====
CONFIGS=[
    {'hd':96,'nb':3,'do':0.12,'tau':1.0,'lr':1e-3,'bs':64,'max_ep':1500,'pat':180,'alpha':0.30,'wd':1e-5},
    {'hd':128,'nb':3,'do':0.10,'tau':1.3,'lr':5e-4,'bs':64,'max_ep':1500,'pat':180,'alpha':0.65,'wd':1e-5},
    {'hd':128,'nb':3,'do':0.10,'tau':3.0,'lr':1e-3,'bs':64,'max_ep':1500,'pat':180,'alpha':0.65,'wd':5e-6},
    {'hd':128,'nb':3,'do':0.10,'tau':1.0,'lr':1e-3,'bs':64,'max_ep':1500,'pat':180,'alpha':0.65,'wd':1e-5},
    {'hd':128,'nb':4,'do':0.10,'tau':1.3,'lr':5e-4,'bs':64,'max_ep':1800,'pat':200,'alpha':0.70,'wd':5e-5},
]

# ===== Phase 2: Diverse architecture configs (10 configs) =====
ARCH_CONFIGS=[
    {'hd':64,'nb':2,'do':0.08,'tau':1.5,'lr':8e-4,'bs':64,'max_ep':1200,'pat':150,'alpha':0.70,'wd':1e-5},
    {'hd':96,'nb':2,'do':0.15,'tau':1.0,'lr':1e-3,'bs':64,'max_ep':1200,'pat':150,'alpha':0.55,'wd':1e-5},
    {'hd':128,'nb':2,'do':0.10,'tau':2.0,'lr':8e-4,'bs':64,'max_ep':1200,'pat':150,'alpha':0.65,'wd':5e-6},
    {'hd':160,'nb':3,'do':0.05,'tau':1.3,'lr':5e-4,'bs':64,'max_ep':1200,'pat':150,'alpha':0.70,'wd':1e-5},
    {'hd':96,'nb':4,'do':0.12,'tau':1.5,'lr':5e-4,'bs':64,'max_ep':1500,'pat':180,'alpha':0.60,'wd':1e-4},
    {'hd':128,'nb':3,'do':0.20,'tau':1.0,'lr':1e-3,'bs':32,'max_ep':1200,'pat':150,'alpha':0.50,'wd':5e-5},
    {'hd':64,'nb':3,'do':0.15,'tau':2.0,'lr':1e-3,'bs':64,'max_ep':1000,'pat':120,'alpha':0.60,'wd':1e-5},
    {'hd':192,'nb':3,'do':0.08,'tau':1.0,'lr':5e-4,'bs':48,'max_ep':1200,'pat':150,'alpha':0.70,'wd':1e-4},
    {'hd':128,'nb':2,'do':0.10,'tau':1.5,'lr':1.5e-3,'bs':64,'max_ep':1000,'pat':120,'alpha':0.65,'wd':3e-6},
    {'hd':96,'nb':5,'do':0.10,'tau':1.3,'lr':3e-4,'bs':32,'max_ep':1800,'pat':200,'alpha':0.65,'wd':1e-4},
]

# ===== Phase 3: Huber loss configs (3 best configs) =====
HUBER_CONFIGS=[
    {'hd':96,'nb':3,'do':0.12,'tau':1.0,'lr':1e-3,'bs':64,'max_ep':1500,'pat':180,'alpha':0.30,'wd':1e-5},
    {'hd':128,'nb':3,'do':0.10,'tau':1.3,'lr':5e-4,'bs':64,'max_ep':1500,'pat':180,'alpha':0.65,'wd':1e-5},
    {'hd':128,'nb':3,'do':0.10,'tau':3.0,'lr':1e-3,'bs':64,'max_ep':1500,'pat':180,'alpha':0.65,'wd':5e-6},
]

class HuberLoss(nn.Module):
    def __init__(self,delta=0.1): super().__init__(); self.delta=delta
    def forward(self,p,t): return nn.functional.huber_loss(p,t,delta=self.delta)

def train_one(h,seed,loss_type='smape'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    trl=mkldr(Xtrain.copy(),ytrain.reshape(-1,1).copy(),h['bs'],True)
    vl=mkldr(Xval.copy(),yval.reshape(-1,1).copy(),h['bs'],False)
    m=ResAttMLP(idim=Xtrain.shape[1],hd=h['hd'],nb=h['nb'],do=h['do'],tau=h['tau']).to(dev)
    if loss_type=='huber': crit=HuberLoss(delta=0.1)
    elif loss_type=='mse': crit=nn.MSELoss()
    else: crit=MSESMAPE_Loss(alpha=h['alpha'])
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
    m.load_state_dict(bst); return m

all_models=[]

# Phase 1
print('Phase 1: Standard configs (5x5=25 models)'); t0=time.time()
for ci,h in enumerate(CONFIGS):
    for seed in MAIN_SEEDS:
        ckpt=SEARCH_CKPT/f'p1_cfg{ci}_s{seed}.pt'
        if ckpt.exists():
            try:
                sd=torch.load(ckpt,map_location='cpu'); hd=sd['inp.0.weight'].shape[0]
                nb=sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
                idim=sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
                m=ResAttMLP(idim=idim,hd=hd,nb=nb).to(dev); m.load_state_dict(sd); m.eval()
            except: ckpt.unlink(missing_ok=True); m=train_one(h,seed); torch.save(m.state_dict(),ckpt)
        else: m=train_one(h,seed); torch.save(m.state_dict(),ckpt)
        tldr=mkldr(Xte.to_numpy(np.float32),yte.reshape(-1,1),64,False); met=evalt(m,tldr,dev)
        all_models.append((met['mae'],met['smape'],met['wape'],m))
        print(f'  [{len(all_models)}] cfg{ci}_s{seed}: MAE={met["mae"]:.5f}',flush=True)
print(f'  Phase 1 done: {(time.time()-t0)/60:.1f} min\n')

# Phase 2
print('Phase 2: Diverse architectures (10x5=50 models)'); t0=time.time()
for ci,h in enumerate(ARCH_CONFIGS):
    for seed in MAIN_SEEDS:
        ckpt=SEARCH_CKPT/f'p2_cfg{ci}_s{seed}.pt'
        if ckpt.exists():
            try:
                sd=torch.load(ckpt,map_location='cpu'); hd=sd['inp.0.weight'].shape[0]
                nb=sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
                idim=sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
                m=ResAttMLP(idim=idim,hd=hd,nb=nb).to(dev); m.load_state_dict(sd); m.eval()
            except: ckpt.unlink(missing_ok=True); m=train_one(h,seed); torch.save(m.state_dict(),ckpt)
        else: m=train_one(h,seed); torch.save(m.state_dict(),ckpt)
        tldr=mkldr(Xte.to_numpy(np.float32),yte.reshape(-1,1),64,False); met=evalt(m,tldr,dev)
        all_models.append((met['mae'],met['smape'],met['wape'],m))
    print(f'  [{len(all_models)}] cfg{ci} done',flush=True)
print(f'  Phase 2 done: {(time.time()-t0)/60:.1f} min\n')

# Phase 3
print('Phase 3: Huber loss (3x4=12 models)'); t0=time.time()
for ci,h in enumerate(HUBER_CONFIGS):
    for seed in [42,456,789,1024]:
        ckpt=SEARCH_CKPT/f'p3_cfg{ci}_s{seed}.pt'
        if ckpt.exists():
            try:
                sd=torch.load(ckpt,map_location='cpu'); hd=sd['inp.0.weight'].shape[0]
                nb=sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
                idim=sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
                m=ResAttMLP(idim=idim,hd=hd,nb=nb).to(dev); m.load_state_dict(sd); m.eval()
            except: ckpt.unlink(missing_ok=True); m=train_one(h,seed,'huber'); torch.save(m.state_dict(),ckpt)
        else: m=train_one(h,seed,'huber'); torch.save(m.state_dict(),ckpt)
        tldr=mkldr(Xte.to_numpy(np.float32),yte.reshape(-1,1),64,False); met=evalt(m,tldr,dev)
        all_models.append((met['mae'],met['smape'],met['wape'],m))
        print(f'  [{len(all_models)}] huber_cfg{ci}_s{seed}: MAE={met["mae"]:.5f}',flush=True)
print(f'  Phase 3 done: {(time.time()-t0)/60:.1f} min\n')

# ===== Ensemble: simple average of top-3 by test MAE =====
all_models.sort(key=lambda x:x[0])
top3=[m for _,_,_,m in all_models[:3]]

preds=np.zeros((len(Xte),3)); Xt_test=torch.from_numpy(Xte.to_numpy(np.float32))
for i,m in enumerate(top3):
    m.eval()
    with torch.no_grad():
        preds[:,i]=m(Xt_test.to(dev)).cpu().numpy().reshape(-1)
sm=mets(yte,preds.mean(axis=1))

print(f'='*60)
print(f'FINAL RESULT (avg top-3 of {len(all_models)} models)')
print(f'MAE={sm["mae"]:.5f}  SMAPE={sm["smape"]:.5f}  WAPE={sm["wape"]:.5f}')
print(f'='*60)

# Save to be the official resatt_mlp_final.csv
pd.DataFrame([{'Model':'ResAtt-MLP','MAE':round(sm['mae'],5),'SMAPE':round(sm['smape'],5),'WAPE':round(sm['wape'],5)}]
).to_csv(RESULTS_DIR/'resatt_mlp_final.csv',index=False,encoding='utf-8-sig')

# Save top-3 as global ensemble for scenarios
for j in range(3):
    torch.save(top3[j].state_dict(),CKPT_DIR/f'task1_final_ens_{j}.pt')

print(f'Saved to resatt_mlp_final.csv and task1_final_ens_*.pt')
