import os,sys,time,json,glob
import numpy as np,soundfile as sf,torch
sys.path.insert(0,'openss')
from model import SpeakerBeamSS
from resemblyzer import VoiceEncoder,preprocess_wav
torch.set_num_threads(2); torch.set_num_interop_threads(1)
m=SpeakerBeamSS().eval(); m.load_state_dict(torch.load('openss-best.pth',map_location='cpu',weights_only=True))
enc=VoiceEncoder(device='cpu'); sr=16000; n=sr*4
spks=sorted(glob.glob('fixtures/LibriSpeech/test-clean/*'))
os.makedirs('results-openss',exist_ok=True)
def sdr(y,x):
 y=y-y.mean(); x=x-x.mean(); p=np.dot(y,x)*x/(np.dot(x,x)+1e-12); return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
def audio(f): return sf.read(f,dtype='float32')[0]
rows=[]
for pair in range(6):
 a,b=spks[pair*2:pair*2+2]; af=sorted(glob.glob(a+'/*/*.flac')); bf=sorted(glob.glob(b+'/*/*.flac'))
 # Choose first sufficiently long utterance; distinct enrollment utterance.
 ai=next(i for i,f in enumerate(af) if len(audio(f))>=n); bi=next(i for i,f in enumerate(bf) if len(audio(f))>=n)
 x=audio(af[ai])[:n]; z=audio(bf[bi])[:n]; x=x/(np.sqrt(np.mean(x*x))+1e-8)*.08; z=z/(np.sqrt(np.mean(z*z))+1e-8)*.08; mix=x+z
 for target,files,idx in [('a',af,ai),('b',bf,bi)]:
  ref=x if target=='a' else z; enroll=audio(files[(idx+1)%len(files)])[:sr*7]
  embed=enc.embed_utterance(preprocess_wav(enroll)); emb=torch.tensor(embed).unsqueeze(0)
  times=[]
  with torch.inference_mode():
   for trial in range(2):
    start=time.perf_counter(); out=m(torch.tensor(mix).reshape(1,1,-1),emb).numpy().reshape(-1); times.append(time.perf_counter()-start)
  out=out[:n]; row=dict(pair=pair,target=target,a=af[ai],b=bf[bi],enrollment=files[(idx+1)%len(files)],input_si_sdr=sdr(mix[:len(out)],ref[:len(out)]),output_si_sdr=sdr(out,ref[:len(out)]),seconds=times,rtf=times[-1]/4)
  row['improvement']=row['output_si_sdr']-row['input_si_sdr']; rows.append(row); print(json.dumps(row),flush=True)
  sf.write(f'results-openss/pair{pair}-{target}-output.wav',out,sr)
 sf.write(f'results-openss/pair{pair}-mix.wav',mix,sr); sf.write(f'results-openss/pair{pair}-a-reference.wav',x,sr);sf.write(f'results-openss/pair{pair}-b-reference.wav',z,sr)
json.dump(rows,open('results-openss/metrics.json','w'),indent=2)
print('summary',np.mean([r['improvement'] for r in rows]),np.mean([r['rtf'] for r in rows]),flush=True)
