"""Bounded resumable CPU runs. No test evaluation until fixed schedule finishes."""
import os,time,json,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import LeNet
R=Path(__file__).resolve().parents[1];D=Path('/tmp/lenet-data');torch.set_num_threads(1);seed=37;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
rows=json.load(open(R/'research/rbf-templates.json'));codes=torch.tensor([[1 if c=='#' else -1 for line in v for c in line] for v in rows]);m=LeNet(codes)
x=torch.from_numpy(np.load(D/'train-images-idx3-ubyte.gz.npy')).float()[:,None];x=F.pad(x,(2,2,2,2))/200-.1;y=torch.from_numpy(np.load(D/'train-labels-idx1-ubyte.gz.npy')).long()
checkpoint=R/'research/checkpoint.pt';rng=np.random.RandomState(seed);order=rng.permutation(len(y));horder=rng.permutation(len(y))[:500];epoch=0;step=0;curvature=None;history=[];epoch_loss=0.;epoch_start=time.time()
if checkpoint.exists():
 c=torch.load(checkpoint,weights_only=True);m.load_state_dict(c['state']);epoch=c['epoch'];step=c['step'];curvature=c['curvature'];history=c['history'];epoch_loss=c['epoch_loss'];order=np.asarray(c['order']);horder=np.asarray(c['horder'])
if (R/'research/failed-run.json').exists():
 print('paused failed-saturation checkpoint; see failed-run.json');raise SystemExit
start=time.time();budget=int(os.environ.get('BUDGET','95'));done=0
# One fixed randomized order is reused every pass, as Appendix C specifies.
while epoch<20 and time.time()-start<budget:
 if curvature is None:
  curvature=[torch.zeros_like(p) for p in m.parameters()];by_id={id(p):i for i,p in enumerate(m.parameters())}
  for ix in horder:
   _,stats=m.stats(x[ix:ix+1],y[ix:ix+1])
   for p,g,h in stats:curvature[by_id[id(p)]]+=h/len(horder)
  print('curvature',epoch,'range',min(h.min().item() for h in curvature),max(h.max().item() for h in curvature),flush=True)
 lr=.0005 if epoch<2 else .0002 if epoch<5 else .0001 if epoch<8 else .00005 if epoch<12 else .00001
 by_id={id(p):i for i,p in enumerate(m.parameters())}
 ix=int(order[step]);loss,stats=m.gradients(x[ix:ix+1],y[ix:ix+1]);epoch_loss+=loss
 with torch.no_grad():
  for p,g in stats:p.addcdiv_(g,curvature[by_id[id(p)]]+.02,value=-lr)
 step+=1;done+=1
 if step==len(order):
  history.append(dict(epoch=epoch+1,train_mse_online=epoch_loss/len(order),learning_rate=lr));print(history[-1],flush=True);epoch+=1;step=0;curvature=None;epoch_loss=0.
c=dict(state=m.state_dict(),epoch=epoch,step=step,curvature=curvature,history=history,epoch_loss=epoch_loss,order=torch.as_tensor(order),horder=torch.as_tensor(horder));torch.save(c,checkpoint);json.dump(dict(epoch=epoch,step=step,complete=epoch==20,history=history,processed_this_run=done,seconds=time.time()-start),open(R/'research/progress.json','w'),indent=2);print('checkpoint',epoch,step,'processed',done,'seconds',time.time()-start,flush=True)
if epoch==20:
 xt=torch.from_numpy(np.load(D/'t10k-images-idx3-ubyte.gz.npy')).float()[:,None];xt=F.pad(xt,(2,2,2,2))/200-.1;yt=torch.from_numpy(np.load(D/'t10k-labels-idx1-ubyte.gz.npy')).long();pred=[]
 with torch.no_grad():
  for i in range(0,len(yt),256):pred.extend(m(xt[i:i+256]).argmin(-1).tolist())
 cm=np.zeros((10,10),int)
 for a,b in zip(yt.tolist(),pred):cm[a,b]+=1
 result=dict(seed=seed,epochs=20,train_samples=len(y),test_samples=len(yt),errors=int((np.array(pred)!=yt.numpy()).sum()),accuracy=float((np.array(pred)==yt.numpy()).mean()),confusion=cm.tolist(),history=history,parameters=sum(p.numel() for p in m.parameters()),criterion='fixed RBF squared Euclidean target distance',optimizer='per-example diagonal Gauss-Newton Levenberg-Marquardt, connection-local curvature, mu .02, 500 samples per pass',codebook='figure3 raster reconstructed approximation; not original digital weights');json.dump(result,open(R/'docs/results.json','w'),indent=2)
