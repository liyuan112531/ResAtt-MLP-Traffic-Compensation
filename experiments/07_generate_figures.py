"""
Generate all paper figures: distribution, comparison, ablation, scenarios, hyperparams.
"""
import os, sys, warnings
os.environ['PYTHONIOENCODING']='utf-8'; warnings.filterwarnings('ignore')
import numpy as np; import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

BASE=Path(__file__).resolve().parent; sys.path.insert(0,str(BASE))
from config import PROCESSED_DIR, RESULTS_DIR, FIG_DIR
FIG_DIR.mkdir(parents=True,exist_ok=True)

# === Publication-quality defaults ===
rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False,
})

COLORS=['#2B579A','#E74856','#107C10','#D83B01','#8661C5','#6B7B8D','#000000']
RESATT_COLOR='#E74856'
RESATT_LIGHT='#D49090'   # 单任务模型——更浅的红褐色，与多任务深红形成对比

# ============================
# Figure 1: Support Rate Distribution
# ============================
modeling=pd.read_csv(PROCESSED_DIR/'modeling_data.csv')
sr=modeling['support_rate']
fig,axes=plt.subplots(1,2,figsize=(8,3.2))
ax=axes[0]; ax.hist(sr,bins=40,color=COLORS[0],edgecolor='white',alpha=0.85)
ax.axvline(sr.mean(),color=RESATT_COLOR,linestyle='--',linewidth=1.5,label=f'Mean={sr.mean():.3f}')
ax.axvline(sr.median(),color='#107C10',linestyle=':',linewidth=1.5,label=f'Median={sr.median():.3f}')
ax.set_xlabel('Support Rate'); ax.set_ylabel('Count'); ax.legend(fontsize=8); ax.set_title('(a) Distribution')
ax=axes[1]
scenarios=[('A: High\n(SR>=0.80)',(sr>=0.8).sum()),('B: Low\n(SR<=0.30)',(sr<=0.3).sum()),
           ('C: Mid\n(0.30~0.80)',((sr>0.3)&(sr<0.8)).sum())]
labels=[s[0] for s in scenarios]; sizes=[s[1] for s in scenarios]
ax.bar(range(3),sizes,color=[COLORS[0],COLORS[1],COLORS[2]])
ax.set_xticks(range(3)); ax.set_xticklabels(labels,fontsize=8)
for i,s in enumerate(sizes): ax.text(i,s+10,str(s),ha='center',fontsize=8)
ax.set_ylabel('Count'); ax.set_title('(b) Scenario Split')
plt.tight_layout(); fig.savefig(FIG_DIR/'fig1_support_rate_distribution.png'); plt.close()
print('Fig 1 saved')

# ============================
# Figure 2: Main Comparison (updated with Multi-Task model)
# ============================
bl=pd.read_csv(RESULTS_DIR/'baselines_comparison.csv')[['Model','MAE','SMAPE','WAPE']]
res=pd.read_csv(RESULTS_DIR/'resatt_mlp_final.csv')[['Model','MAE','SMAPE','WAPE']]
# 将单任务标签改为 "ResAtt-MLP\n(Single-Task)" 以便与多任务对称展示
res['Model']=res['Model'].replace({'ResAtt-MLP':'ResAtt-MLP\n(Single-Task)'})
main=pd.concat([bl,res],ignore_index=True)
# 插入多任务模型结果（合并池 Top-10 集成，来自 09b_combined_ensemble 评估）
mt_row=pd.DataFrame([{'Model':'ResAtt-MLP\n(Multi-Task)','MAE':0.24300,
                       'SMAPE':0.48550,'WAPE':0.39370}])
main=pd.concat([main,mt_row],ignore_index=True)
# 按 MAE 升序排列，多任务模型紧邻单任务
main=main.sort_values('MAE').reset_index(drop=True)

metrics=['MAE','SMAPE','WAPE']
fig,axes=plt.subplots(1,3,figsize=(10.5,3.2))
for i,(metric,ax) in enumerate(zip(metrics,axes)):
    vals=main[metric].values; labels=main['Model'].values
    # 颜色分配：多任务→RESATT_COLOR，单任务→浅红棕，其他→默认色
    colors=[]
    for l in labels:
        if 'Multi' in l:
            colors.append(RESATT_COLOR)
        elif 'ResAtt' in l and 'Single' in l:
            colors.append(RESATT_LIGHT)
        elif 'ResAtt' in l:
            colors.append(RESATT_LIGHT)
        else:
            colors.append(COLORS[i%len(COLORS)])
    bars=ax.barh(range(len(labels)),vals,color=colors,edgecolor='white',height=0.6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=6.5, linespacing=0.8)
    ax.set_xlabel(metric); ax.set_title(f'({chr(97+i)}) {metric}')
    # 标注最优值
    best_idx=np.argmin(vals); ax.text(vals[best_idx]+0.002,best_idx,f'{vals[best_idx]:.5f}',va='center',fontsize=6,fontweight='bold')
plt.tight_layout(); fig.savefig(FIG_DIR/'fig2_model_comparison.png'); plt.close()
print('Fig 2 saved (updated with Multi-Task)')

# ============================
# Figure 3: Ablation
# ============================
abl=pd.read_csv(RESULTS_DIR/'ablation_table.csv')
fig,ax=plt.subplots(figsize=(6.5,2.6))
x=np.arange(len(abl)); w=0.25
for i,(metric,color) in enumerate(zip(['MAE','SMAPE','WAPE'],[COLORS[0],COLORS[1],COLORS[2]])):
    vals=abl[metric].values; offset=(i-1)*w
    bars=ax.bar(x+offset,vals,w,label=metric,color=color,edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels([v.replace('ResAtt-MLP','Full').replace('w/o SMAPE (MSE only)','w/o SMAPE') for v in abl['Variant']],fontsize=7)
ax.legend(fontsize=7); ax.set_ylabel('Error')
for i,metric in enumerate(['MAE','SMAPE','WAPE']):
    full_v=abl[metric].iloc[0]
    ax.axhline(y=full_v,color=COLORS[i],linestyle='--',linewidth=0.7,alpha=0.5)
plt.tight_layout(); fig.savefig(FIG_DIR/'fig3_ablation.png'); plt.close()
print('Fig 3 saved')

# ============================
# Figure 4: Scenarios
# ============================
scen=pd.read_csv(RESULTS_DIR/'scenario_metrics.csv')
fig,ax=plt.subplots(figsize=(5.5,2.8))
x=np.arange(len(scen)); w=0.3
ax.bar(x-w/2,scen['Scen_MAE'],w,label='Scenario Model',color=COLORS[0],edgecolor='white')
ax.bar(x+w/2,scen['Glob_MAE'],w,label='Global Model',color='#CCCCCC',edgecolor='white')
for i in range(len(scen)):
    impr=float(scen['MAE_impr'].iloc[i].replace('%','').replace('+',''))
    ax.text(i,min(scen['Scen_MAE'].iloc[i],scen['Glob_MAE'].iloc[i])-0.015,f'{impr:.0f}%',ha='center',fontsize=8,fontweight='bold',color=RESATT_COLOR)
ax.set_xticks(x); ax.set_xticklabels(scen['Scenario'].str.replace('_',' '),fontsize=8)
ax.legend(fontsize=7); ax.set_ylabel('MAE')
plt.tight_layout(); fig.savefig(FIG_DIR/'fig4_scenarios.png'); plt.close()
print('Fig 4 saved')

# ============================
# Figure 5: Hyperparameter Sensitivity
# ============================
param_files={'hd':'Hidden Dimension','nb':'Residual Blocks','tau':'Temperature tau','do':'Dropout Rate','alpha':'Loss Alpha'}
fig,axes=plt.subplots(2,3,figsize=(8.5,5))
axes=axes.flatten()
for idx,(param,title) in enumerate(param_files.items()):
    ax=axes[idx]
    fpath=RESULTS_DIR/f'hyperparam_{param}.csv'
    if fpath.exists():
        df=pd.read_csv(fpath); cols=df.columns
        x=df[cols[0]].values
        for j,metric in enumerate(['MAE','SMAPE','WAPE']):
            ax.plot(x,df[metric].values,'o-',color=COLORS[j],markersize=4,linewidth=1.2,label=metric)
        ax.set_title(f'({chr(97+idx)}) {title}',fontsize=8)
        ax.set_xlabel(param); ax.legend(fontsize=6)
# hide extra subplot
axes[-1].set_visible(False)
plt.tight_layout(); fig.savefig(FIG_DIR/'fig5_hyperparams.png'); plt.close()
print('Fig 5 saved')

# ============================
# Figure 6: Item-Level Prediction Heterogeneity
# ============================
# 数据：7 个代表性分项的 SMAPE（刚性 3 项 + 弹性 4 项）
# 来自多任务模型合并池 Top-10 集成评估
rigid_items = ['Medical', 'Disability\nCompensation', 'Death\nCompensation']
rigid_smape = [0.5887, 0.6539, 0.4584]
flexible_items = ['Mental\nSolatium', 'Transport', 'Accommodation', 'Other']
flexible_smape = [1.0553, 0.7069, 1.7629, 1.2857]

all_items = rigid_items + flexible_items
all_smape = rigid_smape + flexible_smape
n_rigid = len(rigid_items)
n_flex = len(flexible_items)

fig, ax = plt.subplots(figsize=(6.5, 3.0))
x = np.arange(len(all_items))

# 柱子颜色：刚性项目深蓝，弹性项目深红
bar_colors = [COLORS[0]] * n_rigid + [RESATT_COLOR] * n_flex
bars = ax.bar(x, all_smape, color=bar_colors, edgecolor='white', width=0.55)

# 分组均值水平虚线（仅虚线，无文字标注）
rigid_mean = 0.5670
flexible_mean = 1.2027
ax.axhline(y=rigid_mean, color=COLORS[0], linestyle='--', linewidth=1.2, alpha=0.7)
ax.axhline(y=flexible_mean, color=RESATT_COLOR, linestyle='--', linewidth=1.2, alpha=0.7)

# 在柱子上标注数值
for bar, val in zip(bars, all_smape):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(all_items, fontsize=8, rotation=25, ha='right')
ax.set_ylabel('SMAPE', fontsize=9)
# 标题已由上方图例（Rigid / Flexible Items）替代，此处省略

# 图例（放在图外正上方，含分组均值标注）
legend_elements = [
    Patch(facecolor=COLORS[0], label=f'Rigid Items (mean SMAPE={rigid_mean:.3f})'),
    Patch(facecolor=RESATT_COLOR, label=f'Flexible Items (mean SMAPE={flexible_mean:.3f})'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper center',
          bbox_to_anchor=(0.5, 1.12), ncol=2, framealpha=0.9, edgecolor='#CCCCCC')

ax.set_ylim(0, max(all_smape) * 1.25)  # 顶部留出空间给外部图例
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig6_heterogeneity.png')
plt.close()
print('Fig 6 saved')

print(f'\nAll figures saved to {FIG_DIR}')
