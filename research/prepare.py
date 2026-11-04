from pathlib import Path
import gzip,struct,hashlib,json,urllib.request
import numpy as np,torch
R=Path(__file__).resolve().parents[1];D=Path('/tmp/lenet-data');D.mkdir(exist_ok=True)
expected={'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609', 'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c', 't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6', 't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'}
checks={}
for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']:
 p=D/name
 # URL uses a literal HTTPS mirror and four literal filenames, never user input.
 # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
 if not p.exists():p.write_bytes(urllib.request.urlopen('https://azureopendatastorage.blob.core.windows.net/mnist/'+name,timeout=60).read())
 checks[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 if checks[name]!=expected[name]:raise RuntimeError('Dataset checksum mismatch: '+name)
 raw=gzip.decompress(p.read_bytes());isimage='images' in name;n=struct.unpack('>I',raw[4:8])[0];v=np.frombuffer(raw[16 if isimage else 8:],dtype=np.uint8).copy();np.save(D/(name+'.npy'),v.reshape(n,28,28) if isimage else v)
json.dump(checks,open(R/'docs/data-checksums.json','w'),indent=2)
# Digit RBF templates reconstructed from paper figure 3, raster samples.
# Approximation of printed bitmaps, not recovered original digital codebook.
rows=[['.#####.','.......','..###..','.##.##.','##...##','##...##','##...##','##...##','.##.##.','..###..','.......','.......'],['...##..','..###..','.####..','...##..','...##..','...##..','...##..','...##..','...##..','.######','.......','.......'],['.#####.','.......','.#####.','##...##','#....##','....##.','..###..','.##....','##.....','#######','.......','.......'],['#######','.....##','....##.','...##..','..####.','.....##','.....##','.....##','##...##','.#####.','.......','.......'],['.#####.','.......','.......','.##..##','.##..##','###..##','##...##','##...##','##..###','.######','.....##','.....##'],['.#####.','.......','#######','##.....','##.....','.####..','..####.','.....##','##...##','.#####.','.......','.......'],['..####.','.##....','##.....','##.....','######.','###..##','##...##','##...##','###..##','.#####.','.......','.......'],['#######','.....##','.....##','....##.','...##..','...##..','..##...','..##...','..##...','..##...','.......','.......'],['.#####.','##...##','##...##','##...##','.#####.','##...##','##...##','##...##','##...##','.#####.','.......','.......'],['.#####.','##..###','##...##','##...##','##..###','.######','.....##','.....##','....##.','.####..','.......','.......']]
json.dump(rows,open(R/'research/rbf-templates.json','w'),indent=2)
