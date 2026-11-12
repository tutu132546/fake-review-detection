from pathlib import Path
from detect import frame,D
import numpy as np,pandas as pd,sys,json,time,os,subprocess
from sklearn.neighbors import LocalOutlierFactor
from concurrent.futures import ThreadPoolExecutor
P=D/'cases';P.mkdir(exist_ok=True)
def data(ds):
 f,_=frame(ds);f=f.loc[(f.n_reviews>=3)&(f.burstiness_missing==0)].reset_index(drop=True)
 return f,f[[c for c in f if c.endswith('_z')]].values
def score(x):return -LocalOutlierFactor(n_neighbors=20,algorithm='kd_tree').fit(x).negative_outlier_factor_
def worker(ds,i,t,path):
 sys.path.insert(0,os.environ['DCFO_ADAPTER_DIR'])
 from numeric_modules.optimizationLofProper import OptimizationLOF
 f,x=data(ds);start=time.monotonic()
 o=OptimizationLOF(points=x,k=20,indexStartPoint=i,threshold=t-.0001)
 c=o.findCounterfactual(startPoint=o.startPoint,optimizationFunction=o.optimizationFunctionLOFEasy)
 r={'status':'no_solution'}
 if c is not None and np.isfinite(c).all():
  delta=c-x[i];x[i]=c;s=score(x)[i];r=dict(status='valid' if s<=t else 'not_valid',after=float(s),delta=delta.tolist())
 r['seconds']=time.monotonic()-start;Path(path).write_text(json.dumps(r))
if __name__=='__main__':
 if len(sys.argv)>1:worker(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),sys.argv[4]);sys.exit()
 jobs=[]
 for ds in ['fraudsquad_llama3','fraudsquad_qwen2','fraudsquad_qwen_dsr1']:
  f,x=data(ds);s=score(x);t=float(np.quantile(s[f.cohort=='native'],.95));ids=np.flatnonzero((s>t)&(f.cohort=='hijacked_llm'));ids=sorted(ids,key=lambda i:(-s[i],f.user_id[i]))[:20]
  for i in ids:jobs.append(dict(ds=ds,i=int(i),uid=f.user_id[i],before=float(s[i]),threshold=t))
 (D/'targets.json').write_text(json.dumps(jobs))
 def run(j):
  p=P/(j['ds']+'_'+str(j['i'])+'.json')
  try:
   res=subprocess.run([sys.executable,__file__,j['ds'],str(j['i']),str(j['threshold']),str(p)],timeout=45,capture_output=True)
   if res.returncode:p.write_text(json.dumps({'status':'error','err':res.stderr.decode(errors='replace')[-350:]}))
  except subprocess.TimeoutExpired:p.write_text('{"status":"timeout"}')
  print(j['ds'],j['i'],json.loads(p.read_text())['status'],flush=True)
 with ThreadPoolExecutor(3) as pool:list(pool.map(run,jobs))
