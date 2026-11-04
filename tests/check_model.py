import sys,json,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from model import LeNet,C3
r=Path(__file__).resolve().parents[1];rows=json.load(open(r/'research/rbf-templates.json'));codes=torch.tensor([[1 if c=='#' else -1 for line in v for c in line] for v in rows]);torch.set_num_threads(2);torch.manual_seed(37);m=LeNet(codes)
assert sum(p.numel() for p in m.parameters())==60000
assert sum(len(c) for c in C3)==60
x=torch.randn(1,1,32,32);label=torch.tensor([3]);loss,stats=m.stats(x,label);a=m(x)[0,3];a.backward();assert abs(loss-a.item())<1e-3
for p,g,h in stats:
 assert torch.allclose(g,p.grad,atol=1e-5,rtol=1e-4),(g-p.grad).abs().max()
 assert (h>=0).all() and torch.isfinite(h).all()
print('PASS 60,000 trainable parameters, C3 60 links, manual gradients match autograd, diagonal curvature nonnegative')
t=time.time()
for _ in range(50):m.stats(x,label)
print('seconds/example', (time.time()-t)/50)

loss,grads=m.gradients(x,label)
for p,g in grads:assert torch.allclose(g,p.grad,atol=1e-5,rtol=1e-4)
print("PASS optimized first-order path matches autograd")
