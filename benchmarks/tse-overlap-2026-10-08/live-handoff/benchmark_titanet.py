import os,time,json
import numpy as np,soundfile as sf,torch,sherpa_onnx
from asteroid.models import ConvTasNet
torch.set_num_threads(2);torch.set_num_interop_threads(1)
m=ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k').eval()
cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='titanet.onnx',num_threads=2,provider='cpu'); enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
sr=16000;n=sr*4;os.makedirs('results-titanet',exist_ok=True)
def emb(x):
 s=enc.create_stream();s.accept_waveform(sr,x);s.input_finished();e=np.array(enc.compute(s));return e/(np.linalg.norm(e)+1e-8)
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
metadata=json.load(open('results-openss/metrics.json'));rows=[]
for pair in range(6):
 mix=sf.read(f'results-openss/pair{pair}-mix.wav',dtype='float32')[0];refs=[sf.read(f'results-openss/pair{pair}-{t}-reference.wav',dtype='float32')[0] for t in ['a','b']]
 embeds=[emb(sf.read(r['enrollment'],dtype='float32')[0][:sr*7]) for r in metadata[pair*2:pair*2+2]]
 for win in [4,2,1]:
  outputs=[[],[]];elapsed=0;choices=[];cosines=[];oracle=[[],[]]
  for start in range(0,n,int(win*sr)):
   inp=mix[start:start+int(win*sr)]
   with torch.inference_mode():
    ts=time.perf_counter();out=m(torch.tensor(inp).reshape(1,-1)).numpy()[0];elapsed+=time.perf_counter()-ts
   ts=time.perf_counter();oe=[emb(o) for o in out];elapsed+=time.perf_counter()-ts
   cos=np.array([[np.dot(e,o) for o in oe] for e in embeds]);idx=np.argmax(cos,axis=1);choices.append(idx.tolist());cosines.append(cos.tolist())
   for target in range(2):
    outputs[target].append(out[idx[target]]); oracle[target].append(out[np.argmax([sdr(o,refs[target][start:start+len(o)]) for o in out])])
  for target in range(2):
   y=np.concatenate(outputs[target]);row=dict(pair=pair,target=target,win=win,improvement=sdr(y,refs[target])-sdr(mix,refs[target]),si_sdr=sdr(y,refs[target]),other_si_sdr=sdr(y,refs[1-target]),rtf=elapsed/4,oracle_improvement=sdr(np.concatenate(oracle[target]),refs[target])-sdr(mix,refs[target]),choices=choices,cosines=cosines);rows.append(row);print(json.dumps(row),flush=True);sf.write(f'results-titanet/pair{pair}-{target}-{win}s.wav',y,sr)
json.dump(rows,open('results-titanet/metrics.json','w'),indent=2)
for win in [4,2,1]:
 rr=[r for r in rows if r['win']==win]; print('summary',win,'mean',np.mean([r['improvement'] for r in rr]),'median',np.median([r['improvement'] for r in rr]),'RTF',np.mean([r['rtf'] for r in rr]),'degraded',sum(r['improvement']<0 for r in rr),'wrong',sum(r['si_sdr']<r['other_si_sdr'] for r in rr),'oracle',np.mean([r['oracle_improvement'] for r in rr]),flush=True)
