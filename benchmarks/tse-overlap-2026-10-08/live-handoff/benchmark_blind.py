import os,time,json,glob
import numpy as np,soundfile as sf,torch
from asteroid.models import ConvTasNet
from resemblyzer import VoiceEncoder,preprocess_wav
torch.set_num_threads(2); torch.set_num_interop_threads(1)
m=ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k').eval()
enc=VoiceEncoder(device='cpu'); sr=16000;n=sr*4
os.makedirs('results-blind',exist_ok=True)
def sdr(y,x):
 y=y-y.mean(); x=x-x.mean(); p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
rows=[]
for pair in range(6):
 mix=sf.read(f'results-openss/pair{pair}-mix.wav',dtype='float32')[0]
 refs=[sf.read(f'results-openss/pair{pair}-{t}-reference.wav',dtype='float32')[0] for t in ['a','b']]
 metadata=json.load(open('results-openss/metrics.json'))[pair*2:pair*2+2]
 embeds=[enc.embed_utterance(preprocess_wav(sf.read(r['enrollment'],dtype='float32')[0][:sr*7])) for r in metadata]
 for win in [4,2,1]:
  outputs=[[],[]];total=0;sims=[];choices=[]
  for start in range(0,n,int(win*sr)):
   inp=mix[start:start+int(win*sr)]
   with torch.inference_mode():
    ts=time.perf_counter();out=m(torch.tensor(inp).reshape(1,-1)).numpy()[0];total+=time.perf_counter()-ts
   t=time.perf_counter();oe=[enc.embed_utterance(preprocess_wav(o)) for o in out];total+=time.perf_counter()-t
   cos=np.array([[np.dot(e,o) for o in oe] for e in embeds]);idx=np.argmax(cos,axis=1);sims.append(cos.tolist());choices.append(idx.tolist())
   for target in range(2):outputs[target].append(out[idx[target]])
  for target in range(2):
   y=np.concatenate(outputs[target])[:n];scores=[sdr(y,ref) for ref in refs];row=dict(pair=pair,target=target,win=win,improvement=scores[target]-sdr(mix,refs[target]),si_sdr=scores[target],other_si_sdr=scores[1-target],seconds=total,rtf=total/4,choices=choices,cosines=sims);rows.append(row);print(json.dumps(row),flush=True)
   sf.write(f'results-blind/pair{pair}-{target}-{win}s.wav',y,sr)
json.dump(rows,open('results-blind/metrics.json','w'),indent=2)
for win in [4,2,1]:
 rr=[r for r in rows if r['win']==win];print('summary',win,np.mean([r['improvement'] for r in rr]),np.median([r['improvement'] for r in rr]),np.mean([r['rtf'] for r in rr]),'degraded',sum(r['improvement']<0 for r in rr),'wrong',sum(r['si_sdr']<r['other_si_sdr'] for r in rr),flush=True)
