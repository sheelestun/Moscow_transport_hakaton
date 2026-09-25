"""Эксперимент: помогает ли «физическая» ETA-модель (features/eta_physics.py, cross-fitting по машинам).

Результат (26.09): ETA на чужих машинах ошибается на 33% против 47% у «расстояние/скорость», corr с таргетом 0.19,
но основной модели не помогает: holdout 39.8 -> 40.9, proxy 38.8 -> 38.9, LOVO 78.9 -> 79.2. В модель не добавлено.

Запуск из корня репозитория: python ml/src/experiments/eta_physics_experiment.py
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_catboost as tc
from train_catboost import *
from features.tabular import index_traffic, load_split, clone_sources, FEATURES
from features.eta_physics import track_samples, fit_eta, predict_eta, eta_features_for_points, ETA_FEATURES
t0=time.time()
data=load_all(Path('dataset'))
_,traffic,sched=load_split(Path('dataset'),'train')
src=clone_sources(sched)
TG=index_traffic(traffic)
S=[]
for tid,tg in TG.items():
    s=track_samples(tg)
    if len(s): s['tr_id']=tid; s['group']=src.get(tid,tid); S.append(s)
S=pd.concat(S,ignore_index=True)
print(f'[eta] пар: {len(S)}, машин: {S.tr_id.nunique()}, групп: {S.group.nunique()}, {time.time()-t0:.0f} c', flush=True)
real=sorted(v for v in set(src.values()) if v<9_000_000)
# 1) точность ETA на чужих машинах (по реальным группам)
err_m, err_n = [], []
models={}
for g in real:
    m=fit_eta(S[S.group!=g]); models[g]=m
    h=S[(S.group==g)&(S.tr_id==g)]
    if len(h)==0: continue
    p=predict_eta(m,h); naive=h.dist_m/(h.spd15.clip(lower=3)/3.6)
    err_m+=list(np.abs(p-h.y_s)/h.y_s); err_n+=list(np.abs(naive-h.y_s)/h.y_s)
print(f'[eta] ошибка времени проезда на чужих машинах: модель медиана {np.median(err_m):.0%}, «расстояние/скорость» {np.median(err_n):.0%}', flush=True)
pickle.dump(models, open('ml/artifacts/eta_models_crossfit.pkl','wb'))
# 2) перекрёстные признаки для всех точек
def add(split):
    X,M=data[split]
    E=pd.concat([eta_features_for_points(models[g], X.loc[M.index[M.source==g]]) for g in real if (M.source==g).any()])
    return X.join(E), M
data2={sp: add(sp) for sp in ('train','test','validate')}
print('[eta] признак eta_phys_dev_s: доля NaN', {sp: round(float(data2[sp][0].eta_phys_dev_s.isna().mean()),3) for sp in data2}, flush=True)
X,Mx=data2['train']; print('[eta] corr с таргетом: eta_phys_dev_s', round(X.eta_phys_dev_s.corr(Mx.y),3), '| старый eta_dev_s', round(X.eta_dev_s.corr(Mx.y),3), flush=True)
pickle.dump(data2, open('ml/artifacts/cache_features_eta.pkl','wb'))
FAST={**PARAMS,'iterations':600,'learning_rate':0.07,'depth':8}
F2=FEATURES+['eta_phys_s','eta_phys_dev_s']
for tag,d,F in [('без ETA',data,FEATURES),('с ETA',data2,F2)]:
    ho=eval_holdout(d,F,FAST); px=eval_proxy(d,F,FAST); lv=eval_lovo(d,F,FAST)
    print(f'{tag:8s} holdout {ho["test_mae"]:.2f} | proxy {px["proxy_mae"]:.2f} | LOVO {lv["lovo_mae"]:.2f} (бейзлайн LOVO {lv["lovo_base"]:.2f})  [{time.time()-t0:.0f} c]', flush=True)
print('done')
