"""LeNet-5 architecture and explicit diagonal Gauss-Newton backpropagation.
Paper II.B, Appendix C equations 23,25,27. Fixed RBF MSE criterion (8).
"""
import math,torch
from torch import nn
from torch.nn import functional as F
C3=[(0,1,2),(1,2,3),(2,3,4),(3,4,5),(0,4,5),(0,1,5),(0,1,2,3),(1,2,3,4),(2,3,4,5),(0,3,4,5),(0,1,4,5),(0,1,2,5),(0,1,3,4),(1,2,4,5),(0,2,3,5),(0,1,2,3,4,5)]
def squash(x):return 1.7159*torch.tanh((2/3)*x)
def deriv(y):return (1.7159*2/3)*(1-(y/1.7159)**2)
class Conv(nn.Module):
 def __init__(self,ni,no,k):
  super().__init__();self.w=nn.Parameter(torch.empty(no,ni,k,k));self.b=nn.Parameter(torch.zeros(no));nn.init.uniform_(self.w,-2.4/(ni*k*k),2.4/(ni*k*k))
 def forward(self,x):return squash(F.conv2d(x,self.w,self.b))
 def backward_stats(self,x,y,g,h):
  d=deriv(y);ga=g*d;ha=h*d*d
  gw=torch.nn.grad.conv2d_weight(x,self.w.shape,ga);hw=torch.nn.grad.conv2d_weight(x*x,self.w.shape,ha)
  return F.conv_transpose2d(ga,self.w),F.conv_transpose2d(ha,self.w*self.w),[(self.w,gw,hw),(self.b,ga.sum((0,2,3)),ha.sum((0,2,3)))]
class Pool(nn.Module):
 def __init__(self,n):
  super().__init__();self.w=nn.Parameter(torch.full((1,n,1,1),.25));self.b=nn.Parameter(torch.zeros(1,n,1,1))
 def forward(self,x):return squash(F.avg_pool2d(x,2)*4*self.w+self.b)
 def backward_stats(self,x,y,g,h):
  s=F.avg_pool2d(x,2)*4;d=deriv(y);ga=g*d;ha=h*d*d
  return (ga*self.w).repeat_interleave(2,2).repeat_interleave(2,3),(ha*self.w*self.w).repeat_interleave(2,2).repeat_interleave(2,3),[(self.w,(ga*s).sum((0,2,3),keepdim=True),(ha*(F.avg_pool2d(x*x,2)*4)).sum((0,2,3),keepdim=True)),(self.b,ga.sum((0,2,3),keepdim=True),ha.sum((0,2,3),keepdim=True))]
class Sparse(nn.Module):
 def __init__(self):
  super().__init__();self.w=nn.Parameter(torch.empty(60,5,5));self.b=nn.Parameter(torch.zeros(16));links=[(o,i) for o,ix in enumerate(C3) for i in ix];self.register_buffer('outs',torch.tensor([o for o,i in links]));self.register_buffer('ins',torch.tensor([i for o,i in links]))
  for j,(o,i) in enumerate(links):nn.init.uniform_(self.w[j],-2.4/(len(C3[o])*25),2.4/(len(C3[o])*25))
 def dense(self):
  w=torch.zeros(16,6,5,5,device=self.w.device);w[self.outs,self.ins]=self.w;return w
 def forward(self,x):return squash(F.conv2d(x,self.dense(),self.b))
 def backward_stats(self,x,y,g,h):
  w=self.dense();d=deriv(y);ga=g*d;ha=h*d*d
  gw=torch.nn.grad.conv2d_weight(x,w.shape,ga)[self.outs,self.ins];hw=torch.nn.grad.conv2d_weight(x*x,w.shape,ha)[self.outs,self.ins]
  return F.conv_transpose2d(ga,w),F.conv_transpose2d(ha,w*w),[(self.w,gw,hw),(self.b,ga.sum((0,2,3)),ha.sum((0,2,3)))]
class LeNet(nn.Module):
 def __init__(self,codes,linear=False):
  super().__init__();self.layers=nn.ModuleList([Conv(1,6,5),Pool(6),Sparse(),Pool(16),Conv(16,120,5),Conv(120,84,1)]);self.register_buffer('codes',codes.float());self.linear=linear
  if linear:self.head=nn.Linear(84,10)
 def features(self,x):
  for m in self.layers:x=m(x)
  return x.flatten(1)
 def forward(self,x):
  f=self.features(x)
  return self.head(f) if self.linear else ((f[:,None]-self.codes[None])**2).sum(-1)
 @torch.no_grad()
 def stats(self,x,label):
  states=[x]
  for m in self.layers:states.append(m(states[-1]))
  target=self.codes[label].view(-1,84,1,1);g=2*(states[-1]-target);h=torch.full_like(g,2.);loss=(states[-1]-target).square().sum().item();stats=[]
  for i in range(len(self.layers)-1,-1,-1):g,h,s=self.layers[i].backward_stats(states[i],states[i+1],g,h);stats+=s
  return loss,stats

 @torch.no_grad()
 def gradients(self,x,label):
  states=[x]
  for m in self.layers:states.append(m(states[-1]))
  target=self.codes[label].view(-1,84,1,1);g=2*(states[-1]-target);loss=(states[-1]-target).square().sum().item();stats=[]
  for i in range(len(self.layers)-1,-1,-1):
   m=self.layers[i];v=states[i];z=states[i+1];ga=g*deriv(z)
   if isinstance(m,Pool):
    s=F.avg_pool2d(v,2)*4;stats += [(m.w,(ga*s).sum((0,2,3),keepdim=True)),(m.b,ga.sum((0,2,3),keepdim=True))];g=(ga*m.w).repeat_interleave(2,2).repeat_interleave(2,3)
   else:
    w=m.dense() if isinstance(m,Sparse) else m.w;gw=torch.nn.grad.conv2d_weight(v,w.shape,ga)
    if isinstance(m,Sparse):gw=gw[m.outs,m.ins]
    stats += [(m.w,gw),(m.b,ga.sum((0,2,3)))];g=F.conv_transpose2d(ga,w)
  return loss,stats
