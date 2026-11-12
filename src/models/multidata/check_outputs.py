from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
from detect import D,frame,db
from sklearn.cluster import DBSCAN
rng=np.random.default_rng(11)
x=np.vstack([rng.normal(0,.1,(22,3)),rng.normal(2,.1,(18,3)),np.zeros((8,3)),[[12,3,7]]])
s,noise=db(x);ref=DBSCAN(eps=.5,min_samples=5).fit(x);assert np.array_equal(noise,ref.labels_==-1);assert np.array_equal(noise,s>.5)
from explain import data
jobs=json.loads((D/'targets.json').read_text())
for ds in sorted(set(j['ds'] for j in jobs)):
 f,x=data(ds);js=[j for j in jobs if j['ds']==ds];assert len(js)==20
 for j in js:assert f.iloc[j['i']].user_id==j['uid'] and f.iloc[j['i']].cohort=='hijacked_llm' and j['before']>j['threshold']
d=pd.read_csv(D/'detection.csv');assert len(d)==15 and (d.lo<=d.auc).all() and (d.hi>=d.auc).all()
print('PASS: DBSCAN duplicates/noise, target identities and threshold, 15 AUC intervals')
