from pathlib import Path
import gzip,struct,hashlib,json,urllib.request
import numpy as np,torch
R=Path(__file__).resolve().parents[1];D=Path('/tmp/lenet-data');D.mkdir(exist_ok=True)
checks={}
for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']:
 p=D/name
 # URL uses a literal HTTPS mirror and four literal filenames, never user input.
 # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
 if not p.exists():p.write_bytes(urllib.request.urlopen('https://azureopendatastorage.blob.core.windows.net/mnist/'+name,timeout=60).read())
 checks[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 raw=gzip.decompress(p.read_bytes());isimage='images' in name;n=struct.unpack('>I',raw[4:8])[0];v=np.frombuffer(raw[16 if isimage else 8:],dtype=np.uint8).copy();np.save(D/(name+'.npy'),v.reshape(n,28,28) if isimage else v)
json.dump(checks,open(R/'docs/data-checksums.json','w'),indent=2)
# Digit RBF templates reconstructed from paper figure 3, raster samples.
# Approximation of printed bitmaps, not recovered original digital codebook.
rows=[['.#####.','.......','..###..','.##.##.','##...##','##...##','##...##','##...##','.##.##.','..###..','.......','.......'],['...##..','..###..','.####..','...##..','...##..','...##..','...##..','...##..','...##..','.######','.......','.......'],['.#####.','.......','.#####.','##...##','#....##','....##.','..###..','.##....','##.....','#######','.......','.......'],['#######','.....##','....##.','...##..','..####.','.....##','.....##','.....##','##...##','.#####.','.......','.......'],['.#####.','.......','.......','.##..##','.##..##','###..##','##...##','##...##','##..###','.######','.....##','.....##'],['.#####.','.......','#######','##.....','##.....','.####..','..####.','.....##','##...##','.#####.','.......','.......'],['..####.','.##....','##.....','##.....','######.','###..##','##...##','##...##','###..##','.#####.','.......','.......'],['#######','.....##','.....##','....##.','...##..','...##..','..##...','..##...','..##...','..##...','.......','.......'],['.#####.','##...##','##...##','##...##','.#####.','##...##','##...##','##...##','##...##','.#####.','.......','.......'],['.#####.','##..###','##...##','##...##','##..###','.######','.....##','.....##','....##.','.####..','.......','.......']]
json.dump(rows,open(R/'research/rbf-templates.json','w'),indent=2)
