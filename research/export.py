"""Export the completed local Adam RBF checkpoint to transparent JSON shards."""
import json,torch,hashlib
from pathlib import Path
from model import LeNet,C3
r=Path(__file__).resolve().parents[1]
a=json.load(open(r/'research/rbf-templates.json'));codes=torch.tensor([[1 if c=='#' else -1 for row in x for c in row] for x in a]);m=LeNet(codes);m.load_state_dict(torch.load(r/'research/ablation-rbf_mse.pt',weights_only=True));shards=[]
for key,v in m.state_dict().items():
 if key in ['layers.2.outs','layers.2.ins']:continue
 values=v.detach().flatten().tolist()
 for i in range(0,len(values),8000):
  name='weights-'+str(len(shards))+'.json';json.dump(values[i:i+8000],open(r/'docs'/name,'w'),separators=(',',':'));shards.append(dict(file=name,key=key,offset=i,sha256=hashlib.sha256((r/'docs'/name).read_bytes()).hexdigest()))
json.dump(dict(c3=C3,shards=shards),open(r/'docs/model-index.json','w'),separators=(',',':'))
