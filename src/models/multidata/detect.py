from pathlib import Path
import numpy as np,pandas as pd,zipfile,io,hashlib,json,sys,warnings
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor,KDTree
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import os
D=Path(os.environ.get('C_RUN_DIR','data/processed/c_nov12')).resolve();D.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cohort_uncertainty import bootstrap_auc
Z=zipfile.ZipFile(os.environ['C_FEATURE_ZIP'])
def frame(ds):
 b=Z.read('features/feature_table_'+ds+'.csv'); f=pd.read_csv(io.BytesIO(b),dtype={'user_id':str}).sort_values('user_id',kind='stable').reset_index(drop=True)
 return f,hashlib.sha256(b).hexdigest()
def db(x):
 v,idx,w=np.unique(x,axis=0,return_inverse=True,return_counts=True); model=DBSCAN(eps=.5,min_samples=5,algorithm='kd_tree',n_jobs=1).fit(v,sample_weight=w)
 core=v[model.core_sample_indices_];dist=KDTree(core).query(x,k=1)[0].ravel() if len(core) else np.ones(len(x))
 return dist,model.labels_[idx]==-1
if __name__=='__main__':
 rows=[]
 for ds in ['yelpchi','fraudsquad_llama3','fraudsquad_qwen2','fraudsquad_qwen_dsr1','yelpzip']:
  f,h=frame(ds);f=f.loc[f.cohort!='self_injected']; x=f[[c for c in f if c.endswith('_z')]+['burstiness_missing']].values;y=f.account_is_fake.values
  lof=-LocalOutlierFactor(n_neighbors=20,algorithm='kd_tree',n_jobs=2).fit(x).negative_outlier_factor_
  iff=-IsolationForest(n_estimators=200,max_samples=256,random_state=42,n_jobs=2).fit(x).score_samples(x)
  dist,noise=db(x);S=np.column_stack([lof,iff,dist]);ci=np.quantile(bootstrap_auc(y,S,1000,42),[.025,.975],axis=0)
  for j,m in enumerate(['LOF','iForest','DBSCAN-core-distance']):
   s=S[:,j];o=np.argsort(-s,kind='stable')[:50];cut=s[o[-1]]; rows.append(dict(dataset=ds,method=m,n=len(y),positives=sum(y),auc=roc_auc_score(y,s),lo=ci[0,j],hi=ci[1,j],p50=y[o].mean(),boundary_ties=sum(s==cut),sha256=h))
  pd.DataFrame(rows).to_csv(D/'detection.csv',index=False);print(ds,'OK',flush=True)
