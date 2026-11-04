"""Controlled output criterion/head package comparison under the same Adam protocol.
Not historical training and not an isolated head-only causal effect.
"""
from pathlib import Path
import json,time,torch,numpy as np
from torch.nn import functional as F
from model import LeNet
R=Path(__file__).resolve().parents[1];D=Path('/tmp/lenet-data');torch.set_num_threads(2);seed=37
rows=json.load(open(R/'research/rbf-templates.json'));codes=torch.tensor([[1 if c=='#' else -1 for line in v for c in line] for v in rows]);x=torch.from_numpy(np.load(D/'train-images-idx3-ubyte.gz.npy')).float()[:,None];x=F.pad(x,(2,2,2,2))/200-.1;y=torch.from_numpy(np.load(D/'train-labels-idx1-ubyte.gz.npy')).long();xt=torch.from_numpy(np.load(D/'t10k-images-idx3-ubyte.gz.npy')).float()[:,None];xt=F.pad(xt,(2,2,2,2))/200-.1;yt=torch.from_numpy(np.load(D/'t10k-labels-idx1-ubyte.gz.npy')).long();out={}
for linear in [False,True]:
 
 if (R/'docs/ablation.json').exists():
  old=json.load(open(R/'docs/ablation.json'))['comparisons'];out.update(old)
  if ('linear_ce' if linear else 'rbf_mse') in old:continue
 torch.manual_seed(seed);np.random.seed(seed);m=LeNet(codes,linear);opt=torch.optim.Adam(m.parameters(),lr=.001);gen=torch.Generator().manual_seed(seed);history=[];start=time.time()
 for epoch in range(10):
  m.train();order=torch.randperm(len(y),generator=gen);total=0.
  for i in range(0,len(order),128):
   ix=order[i:i+128];opt.zero_grad();z=m(x[ix]);loss=F.cross_entropy(z,y[ix]) if linear else z[torch.arange(len(ix)),y[ix]].mean()/84;loss.backward();opt.step();total+=loss.item()*len(ix)
  history.append(total/len(y));print('linear' if linear else 'rbf',epoch+1,history[-1],round(time.time()-start,2),flush=True)
 m.eval();pred=[]
 with torch.no_grad():
  for i in range(0,len(yt),256):
   z=m(xt[i:i+256]);pred.extend((z.argmax(-1) if linear else z.argmin(-1)).tolist())
 cm=np.zeros((10,10),int)
 for a,b in zip(yt.tolist(),pred):cm[a,b]+=1
 name='linear_ce' if linear else 'rbf_mse';torch.save(m.state_dict(),R/'research'/('ablation-'+name+'.pt'));out[name]=dict(errors=int((np.array(pred)!=yt.numpy()).sum()),accuracy=float((np.array(pred)==yt.numpy()).mean()),confusion=cm.tolist(),history=history,seconds=time.time()-start,parameters=sum(p.numel() for p in m.parameters()))
 json.dump(dict(seed=seed,epochs=10,optimizer='Adam',learning_rate=.001,batch_size=128,augmentation=False,train_samples=60000,test_samples=10000,protocol='fixed schedule; no test-driven tuning',comparisons=out),open(R/'docs/ablation.json','w'),indent=2)
