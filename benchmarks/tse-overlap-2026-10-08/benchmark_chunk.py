import sys,json,time
import numpy as np,soundfile as sf,torch
sys.path.insert(0,'openss')
from model import SpeakerBeamSS
from resemblyzer import VoiceEncoder,preprocess_wav
torch.set_num_threads(2); torch.set_num_interop_threads(1);m=SpeakerBeamSS().eval();m.load_state_dict(torch.load('openss-best.pth',map_location='cpu',weights_only=True));enc=VoiceEncoder(device='cpu');sr=16000
rows=[]
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
metadata=json.load(open('results-openss/metrics.json'))
for pair in range(6):
 mix=sf.read(f'results-openss/pair{pair}-mix.wav',dtype='float32')[0]
 for target,t in enumerate(['a','b']):
  ref=sf.read(f'results-openss/pair{pair}-{t}-reference.wav',dtype='float32')[0];enroll=sf.read(metadata[pair*2+target]['enrollment'],dtype='float32')[0][:sr*7]; emb=torch.tensor(enc.embed_utterance(preprocess_wav(enroll))).reshape(1,-1)
  for win in [2,1,.5]:
   pieces=[];elapsed=0
   for start in range(0,len(mix),int(sr*win)):
    x=mix[start:start+int(sr*win)]
    with torch.inference_mode():
     ts=time.perf_counter();out=m(torch.tensor(x).reshape(1,1,-1),emb).numpy().reshape(-1);elapsed+=time.perf_counter()-ts
    pieces.append(out)
   y=np.concatenate(pieces); row=dict(pair=pair,target=t,win=win,si_sdr=sdr(y,ref[:len(y)]),improvement=sdr(y,ref[:len(y)])-sdr(mix[:len(y)],ref[:len(y)]),rtf=elapsed/4); rows.append(row);print(json.dumps(row),flush=True)
json.dump(rows,open('results-openss/chunk-metrics.json','w'),indent=2)
for w in [2,1,.5]:
 rr=[r for r in rows if r['win']==w];print('summary',w,np.mean([r['improvement'] for r in rr]),np.median([r['improvement'] for r in rr]),np.mean([r['rtf'] for r in rr]),sum(r['improvement']<0 for r in rr))
