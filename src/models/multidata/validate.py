from detect import frame,db,Z,D
from explain import data,score,P
from sklearn.ensemble import IsolationForest
import numpy as np,pandas as pd,json,zipfile,hashlib
RAW=['n_reviews','active_days','daily_avg_reviews','max_daily_reviews','burstiness','rating_deviation','extreme_ratio']
SETS={'llama3':'llama3_8b','qwen2':'qwen_72b','qwen_dsr1':'qwen_dsr1'}
SHAS=['267fd8efdeec07a29de79615453bdb9a7b364f6b','b7f3527f5dd4363c891b88af67dd6510d577e033','742792c762e3d1fc47af05d9a4083505101cf051','5d97703d89e5b1aa976b16e44f0f26c4d91f3c15']
def events(name,sha):
 import os
 from pathlib import Path
 path=next(p for p in [Path(os.getenv('C_AMAZON_DIR',str(D/'upstream/data/Amazon')))/name,D/('real.zip' if name.endswith('.zip') else name)] if p.exists())
 b=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==sha
 source=zipfile.ZipFile(path).open('amazon_2022_5core.csv') if name.endswith('.zip') else path
 f=pd.read_csv(source,usecols=['user','product','star','reviewTime'],dtype={'user':str,'product':str})
 f['dt']=pd.to_datetime(f.reviewTime,unit='s');return f
def agg(f):
 f=f.sort_values(['user','dt']).copy();day=f.dt.dt.normalize();g=f.groupby('user');a=g.size().to_frame('n_reviews');a['active_days']=(day.groupby(f.user).max()-day.groupby(f.user).min()).dt.days+1
 a['daily_avg_reviews']=a.n_reviews/a.active_days;a['max_daily_reviews']=f.groupby([f.user,day]).size().groupby(level=0).max()
 gap=g.dt.diff().dt.total_seconds().groupby(f.user);a['burstiness']=(gap.std(ddof=0)/gap.mean()).where(a.n_reviews>=3)
 a['rating_deviation']=(f.star-f.groupby('product').star.transform('mean')).groupby(f.user).mean();a['extreme_ratio']=f.star.isin([1,5]).groupby(f.user).mean();return a[RAW]
def run():
 real=events('amazon_2022_5core.csv.zip',SHAS[0]);base=agg(real);jobs=json.loads((D/'targets.json').read_text());summ=[];details=[];metrics=[]
 for k,(name,src) in enumerate(SETS.items()):
  ds='fraudsquad_'+name;gen=events('reviews_from_'+src+'.csv',SHAS[k+1]);f,x=data(ds);a=agg(pd.concat([real,gen],ignore_index=True)).loc[f.user_id];a0=base.loc[f.user_id]
  dif=np.max(abs(a.values-f[RAW].values),axis=0);print(ds,'reconstruction errors',dif,flush=True);assert np.all(dif<1e-5),dif
  sc=json.loads(Z.read('features/scaler_'+ds+'.json'))['features'];sd=np.array([sc[c]['std'] for c in RAW]);mean=np.array([sc[c]['mean'] for c in RAW]);labels=f.cohort.eq('native').values
  edit=gen.copy(); edits=0; selected=[];local=[]
  for j in (v for v in jobs if v['ds']==ds):
   i=j['i'];assert f.user_id[i]==j['uid'];selected.append(i);r=json.loads((P/(ds+'_'+str(i)+'.json')).read_text());row=dict(dataset=ds,user=j['uid'],status=r['status'],before=j['before'],threshold=j['threshold'],after=r.get('after'),action='none')
   if r['status']=='valid':
    delta=np.array(r['delta']);truth=(a0.iloc[i].values-a.iloc[i].values)/sd;top=int(np.argmax(abs(delta)));good=np.isfinite(truth)
    top_truth=np.argmax(np.where(good,abs(truth),-1));row.update(top=RAW[top],truth_top=RAW[top_truth],match=bool(top==top_truth),signed=bool(good[top] and delta[top]*truth[top]>0),cos=float(np.dot(delta[good],truth[good])/(np.linalg.norm(delta[good])*np.linalg.norm(truth[good]))) if np.linalg.norm(truth[good])*np.linalg.norm(delta[good]) else None,truth_dims=int(good.sum()))
    # Offline, single prescribed edit to synthetic rows only; never remove reviews.
    uid=j['uid'];ix=edit.index[edit.user.eq(uid)];hist=real[real.user.eq(uid)].sort_values('dt');assert 1<=len(ix)<=2
    if top in (5,6):
     edit.loc[ix,'star']=float(hist.star.mode().iloc[0]);row['action']='historical_modal_rating'
    elif top in (1,2,3,4):
     v=hist.dt.iloc[[len(hist)//3,2*len(hist)//3][:len(ix)]].values;edit.loc[ix,'dt']=v;row['action']='historical_time_quantiles'
    row['changed_events']=int((edit.loc[ix,'star'].ne(gen.loc[ix,'star'])|edit.loc[ix,'dt'].ne(gen.loc[ix,'dt'])).sum());edits+=int(row['changed_events']>0)
   local.append(row)
  trial=agg(pd.concat([real,edit],ignore_index=True)).loc[f.user_id]
  for row,i,j in zip(local,selected,[j for j in jobs if j['ds']==ds]):
   if not row.get('changed_events'):continue
   top=row['top']; delta=json.loads((P/(ds+'_'+str(i)+'.json')).read_text())['delta'][RAW.index(top)]
   if not ((trial.iloc[i][top]-a.iloc[i][top])*delta>0):
    ix=edit.index[edit.user.eq(row['user'])];edit.loc[ix,['star','dt']]=gen.loc[ix,['star','dt']];row['action']='no_matching_direction';row['changed_events']=0;edits-=1
  post=agg(pd.concat([real,edit],ignore_index=True)).loc[f.user_id]
  for row,i,j in zip(local,selected,[j for j in jobs if j['ds']==ds]):
   if row.get('changed_events'):
    top=row['top'];delta=json.loads((P/(ds+'_'+str(i)+'.json')).read_text())['delta'][RAW.index(top)]
    row['realized_direction']=bool((post.iloc[i][top]-a.iloc[i][top])*delta>0)
  X=x+(post.values-a.values)/sd
  # Preserve original evaluation cohort. Undefined post-edit time variability is imputed and flagged explicitly.
  X0=np.column_stack([x,np.zeros(len(x))]);miss=np.isnan(X[:,4]);full,_=frame(ds);X[miss,4]=full.loc[full.burstiness_missing.eq(1),'burstiness_z'].iloc[0];X=np.column_stack([X,miss.astype(float)])
  def allscores(mat):return [score(mat),-IsolationForest(n_estimators=200,max_samples=256,random_state=42,n_jobs=2).fit(mat).score_samples(mat),db(mat)[0]]
  orig=allscores(X0);new=allscores(X)
  for m,b,c in zip(['LOF','iForest','DBSCAN-core-distance'],orig,new):
   t=np.quantile(b[labels],.95);pre=b[selected]>t;po=c[selected]>t
   changed=np.array([row.get('changed_events',0)>0 for row in local]);metrics.append(dict(direction_confirmed=sum(row.get('realized_direction',False) for row in local),changed_pre=sum(pre&changed),changed_post=sum(po&changed),changed_retained=sum(pre&po&changed),dataset=ds,method=m,threshold=t,selected=len(selected),actions=edits,pre=sum(pre),post=sum(po),still_detected=sum(pre&po),new_flags=sum(~pre&po),post_missing=sum(miss)))
  details+=local
  valid=[r for r in local if r['status']=='valid'];summ.append(dict(dataset=ds,selected=len(local),valid=len(valid),top1=sum(r['match'] for r in valid),direction=sum(r['signed'] for r in valid),mean_cos=np.mean([r['cos'] for r in valid if r['cos'] is not None]),partial_truth=sum(r['truth_dims']!=7 for r in valid)))
  pd.DataFrame(summ).to_csv(D/'fidelity.csv',index=False);pd.DataFrame(metrics).to_csv(D/'robustness.csv',index=False);pd.DataFrame(details).to_csv(D/'audit_local.csv',index=False)
if __name__=='__main__':run()
