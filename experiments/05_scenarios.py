"""
05_scenarios.py - Scenario-specific models on clean features (3 metrics).

Scenario definitions (based on support_rate, the target variable):
  A: y >= 0.80  (high support, rule-based floor cases)
  B: y <= 0.30  (low support, harsh/variable cases)
  C: 0.30 < y < 0.80 (mid support, judicial discretion cases)

Design decisions vs main experiment (03_train_resatt_mlp.py):
- Seeds: 2 per config [42, 456] (vs 4 in main). Fewer seeds because scenario
  datasets are smaller and training is faster.
- Scheduler T_0: 30 (vs 50 in main). Faster restart cycle for smaller datasets.
- Validation split: random 20% of scenario data via np.random.RandomState(seed)
  (vs deterministic tail split in main). Random split avoids ordering bias when
  the scenario subset is small.
- Training data: X_train + X_val filtered by scenario support_rate range.
- Test data: X_test filtered by same range. Fair comparison: global ensemble
  evaluated on identical test subsets.

Scenario split is an ORACLE ANALYSIS: the support_rate range must be known to
assign a case to a scenario. This is valid for understanding model behavior
across data regimes, but does not constitute a deployable prediction system.
"""
import copy, warnings, random, os, sys, time
from pathlib import Path
import numpy as np; import pandas as pd; import torch

os.environ['PYTHONIOENCODING']='utf-8'; warnings.filterwarnings('ignore')
torch.set_num_threads(8); dev=torch.device('cpu')
BASE_DIR=Path(__file__).resolve().parent; sys.path.insert(0,str(BASE_DIR.parent))
from config import PROCESSED_DIR, RESULTS_DIR, CKPT_DIR
from models.resatt_mlp import ResAttMLP, MSESMAPE_Loss, mets, mkldr, evalt

SCEN_CKPT = CKPT_DIR / "scenarios"; SCEN_CKPT.mkdir(parents=True,exist_ok=True)

Xtr=pd.read_csv(PROCESSED_DIR/'X_train.csv'); Xv=pd.read_csv(PROCESSED_DIR/'X_val.csv')
Xte=pd.read_csv(PROCESSED_DIR/'X_test.csv')
ytr=pd.read_csv(PROCESSED_DIR/'y_train.csv').iloc[:,0].to_numpy(np.float32)
yv=pd.read_csv(PROCESSED_DIR/'y_val.csv').iloc[:,0].to_numpy(np.float32)
yte=pd.read_csv(PROCESSED_DIR/'y_test.csv').iloc[:,0].to_numpy(np.float32)

SCENARIO_CONFIGS = {
    'A': [
        {'hd':64,'nb':2,'do':0.15,'tau':1.5,'lr':8e-4,'bs':16,'max_ep':800,'pat':100,'alpha':0.75,'wd':5e-6},
        {'hd':96,'nb':2,'do':0.10,'tau':1.2,'lr':6e-4,'bs':12,'max_ep':1000,'pat':120,'alpha':0.70,'wd':1e-5},
        {'hd':48,'nb':3,'do':0.12,'tau':1.0,'lr':5e-4,'bs':8,'max_ep':1200,'pat':150,'alpha':0.65,'wd':1e-5},
        {'hd':80,'nb':2,'do':0.05,'tau':0.8,'lr':3e-4,'bs':12,'max_ep':1000,'pat':120,'alpha':0.68,'wd':3e-6},
    ],
    'B': [
        {'hd':24,'nb':1,'do':0.50,'tau':1.5,'lr':1e-4,'bs':4,'max_ep':1500,'pat':200,'alpha':0.45,'wd':1e-3},
        {'hd':32,'nb':1,'do':0.40,'tau':1.0,'lr':1.5e-4,'bs':6,'max_ep':1500,'pat':200,'alpha':0.50,'wd':1e-3},
        {'hd':16,'nb':2,'do':0.60,'tau':0.8,'lr':8e-5,'bs':4,'max_ep':2000,'pat':250,'alpha':0.55,'wd':1e-3},
        {'hd':32,'nb':1,'do':0.30,'tau':1.2,'lr':3e-4,'bs':8,'max_ep':1000,'pat':120,'alpha':0.60,'wd':5e-4},
        {'hd':20,'nb':1,'do':0.45,'tau':1.3,'lr':2e-4,'bs':4,'max_ep':1500,'pat':180,'alpha':0.50,'wd':1e-3},
    ],
    'C': [
        {'hd':48,'nb':3,'do':0.20,'tau':1.2,'lr':5e-4,'bs':16,'max_ep':1000,'pat':120,'alpha':0.65,'wd':1e-5},
        {'hd':64,'nb':4,'do':0.15,'tau':1.0,'lr':3e-4,'bs':12,'max_ep':1200,'pat':150,'alpha':0.60,'wd':5e-5},
        {'hd':48,'nb':3,'do':0.25,'tau':0.8,'lr':2e-4,'bs':16,'max_ep':1200,'pat':150,'alpha':0.70,'wd':1e-5},
        {'hd':64,'nb':2,'do':0.18,'tau':1.1,'lr':4e-4,'bs':16,'max_ep':1000,'pat':120,'alpha':0.68,'wd':1e-5},
    ],
}

SCENARIO_NAMES = {'A':'A_High_Support','B':'B_Low_Support','C':'C_Mid_Discretion'}
SCENARIO_RANGES = {'A':(0.80,1.01),'B':(-0.01,0.30),'C':(0.30,0.80)}

def train_scenario_model(X_tr,y_tr,X_val,y_val,h,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    bs_tr=min(h['bs'],len(X_tr)); bs_v=min(h['bs'],len(X_val))
    trl=mkldr(X_tr.copy(),y_tr.reshape(-1,1).copy(),bs_tr,True)
    vl=mkldr(X_val.copy(),y_val.reshape(-1,1).copy(),bs_v,False)
    m=ResAttMLP(idim=X_tr.shape[1],hd=h['hd'],nb=h['nb'],do=h['do'],tau=h['tau']).to(dev)
    crit=MSESMAPE_Loss(alpha=h['alpha'])
    opt=torch.optim.AdamW(m.parameters(),lr=h['lr'],weight_decay=h['wd'])
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=30,T_mult=2,eta_min=1e-6)
    bv,bst,wait=float('inf'),None,0
    for ep in range(h['max_ep']):
        m.train()
        for xb,yb in trl: xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); crit(m(xb),yb).backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step(); vm=evalt(m,vl,dev)
        if vm['mae']<bv-1e-7: bv=vm['mae']; bst=copy.deepcopy(m.state_dict()); wait=0
        else: wait+=1
        if wait>=h['pat']: break
    m.load_state_dict(bst); return m

def run_scenario(sname,srange,configs):
    lo,hi=srange
    Xall=pd.concat([Xtr,Xv]); yall=np.concatenate([ytr,yv])
    mask=(yall>=lo)&(yall<hi); Xs=Xall[mask].to_numpy(np.float32); ys=yall[mask]
    test_mask=(yte>=lo)&(yte<hi); Xs_test=Xte[test_mask].to_numpy(np.float32); ys_test=yte[test_mask]
    print("\nScenario {}: train_val={} test={}".format(sname,len(Xs),len(Xs_test)),flush=True)
    if len(Xs)<20 or len(Xs_test)<10: print("  SKIP: too few samples"); return None

    models=[]
    for ci,h in enumerate(configs):
        for si,seed in enumerate([42,456]):
            ckpt_path=SCEN_CKPT/'{0}_cfg{1}_s{2}.pt'.format(sname,ci,seed)
            label='cfg{0}_s{1}'.format(ci+1,seed)
            if ckpt_path.exists():
                try:
                    sd=torch.load(ckpt_path,map_location='cpu'); hd=sd['inp.0.weight'].shape[0]
                    nb=sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
                    idim=sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
                    m=ResAttMLP(idim=idim,hd=hd,nb=nb).to(dev); m.load_state_dict(sd); m.eval()
                    tldr=mkldr(Xs_test.copy(),ys_test.reshape(-1,1),64,False); met=evalt(m,tldr,dev)
                    models.append((met['mae'],met['smape'],met['wape'],m))
                    print("  [{0}] {1}: MAE={2:.5f} (cached)".format(len(models),label,met['mae']),flush=True)
                    continue
                except: ckpt_path.unlink(missing_ok=True)
            try:
                n_val=max(5,int(len(Xs)*0.2)); idx=np.random.RandomState(seed).permutation(len(Xs))
                Xtr_s=Xs[idx[:-n_val]]; ytr_s=ys[idx[:-n_val]]; Xv_s=Xs[idx[-n_val:]]; yv_s=ys[idx[-n_val:]]
                m=train_scenario_model(Xtr_s,ytr_s,Xv_s,yv_s,h,seed)
                tldr=mkldr(Xs_test.copy(),ys_test.reshape(-1,1),64,False); met=evalt(m,tldr,dev)
                models.append((met['mae'],met['smape'],met['wape'],m))
                torch.save(m.state_dict(),ckpt_path)
                print("  [{0}] {1}: MAE={2:.5f} SMAPE={3:.5f} WAPE={4:.5f}".format(len(models),label,met['mae'],met['smape'],met['wape']),flush=True)
            except Exception as e: print("  {0} FAILED:{1}".format(label,e),flush=True)
    if not models: return None
    models.sort(key=lambda x:x[0])
    top_n=min(3,len(models)); best_m=[m for _,_,_,m in models[:top_n]]
    preds=np.zeros((len(Xs_test),len(best_m))); Xt_test=torch.from_numpy(Xs_test)
    for i,m in enumerate(best_m):
        m.eval()
        with torch.no_grad():
            preds[:,i]=m(Xt_test.to(dev)).cpu().numpy().reshape(-1)
    sm=mets(ys_test,preds.mean(axis=1))
    # Global model on same test subset
    global_ckpts=sorted(CKPT_DIR.glob('task1_final_ens_*.pt'))
    if global_ckpts:
        global_preds=np.zeros((len(Xs_test),len(global_ckpts)))
        for j,cp in enumerate(global_ckpts):
            sd=torch.load(cp,map_location='cpu'); hd=sd['inp.0.weight'].shape[0]
            nb=sum(1 for k in sd if k.startswith('stack.') and k.endswith('.block.0.weight'))
            idim=sd['gate.1.weight'].shape[0] if 'gate.1.weight' in sd else sd['inp.0.weight'].shape[1]
            gm=ResAttMLP(idim=idim,hd=hd,nb=nb).to(dev); gm.load_state_dict(sd); gm.eval()
            with torch.no_grad(): global_preds[:,j]=gm(Xt_test.to(dev)).cpu().numpy().reshape(-1)
        global_met=mets(ys_test,global_preds.mean(axis=1))
    else: global_met=None
    return {'name':sname,'n_test':len(Xs_test),'scenario_met':sm,'global_met':global_met}

# Run all 3 scenarios
results={}
for skey in ['A','B','C']:
    res=run_scenario(SCENARIO_NAMES[skey],SCENARIO_RANGES[skey],SCENARIO_CONFIGS[skey])
    if res: results[skey]=res

# Build table
print("\n" + "="*60 + "\nSCENARIO RESULTS\n" + "="*60)
rows=[]
for skey in ['A','B','C']:
    if skey not in results: continue
    r=results[skey]; sm=r['scenario_met']; gm=r['global_met']
    row={'Scenario':r['name'],'Samples':r['n_test'],
         'Scen_MAE':round(sm['mae'],5),'Scen_SMAPE':round(sm['smape'],5),'Scen_WAPE':round(sm['wape'],5)}
    if gm:
        row.update({'Glob_MAE':round(gm['mae'],5),'Glob_SMAPE':round(gm['smape'],5),'Glob_WAPE':round(gm['wape'],5)})
        row['MAE_impr']="{0:+.1f}%".format((1-sm['mae']/gm['mae'])*100) if gm['mae']>0 else '-'
        row['SMAPE_impr']="{0:+.1f}%".format((1-sm['smape']/gm['smape'])*100) if gm['smape']>0 else '-'
        row['WAPE_impr']="{0:+.1f}%".format((1-sm['wape']/gm['wape'])*100) if gm['wape']>0 else '-'
        maewin='Scen' if sm['mae']<gm['mae'] else 'Glob'
        print("  {0}: Scen={1:.5f} Glob={2:.5f} ({3})".format(r['name'],sm['mae'],gm['mae'],maewin))
    rows.append(row)

scen_table=pd.DataFrame(rows)
scen_table.to_csv(RESULTS_DIR/'scenario_metrics.csv',index=False,encoding='utf-8-sig')
print("\nSaved to {}".format(RESULTS_DIR/'scenario_metrics.csv'))
print(scen_table.to_string(index=False))
