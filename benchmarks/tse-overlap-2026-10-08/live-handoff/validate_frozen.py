# Fresh deterministic evaluation after calibration investigation. No tuning on this split.
import glob,time,json,os
import numpy as np,soundfile as sf,sherpa_onnx,onnxruntime as ort
sr=16000;n=sr*4
cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='titanet.onnx',num_threads=2,provider='cpu');enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;sep=ort.InferenceSession('convtasnet-2s-safe.onnx',o,providers=['CPUExecutionProvider'])
def emb(x):
 s=enc.create_stream();s.accept_waveform(sr,x);s.input_finished();e=np.array(enc.compute(s));return e/(np.linalg.norm(e)+1e-8)
def aud(f):return sf.read(f,dtype='float32')[0]
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
spks=sorted(glob.glob('fixtures/LibriSpeech/test-clean/*'));rng=np.random.default_rng(91008);rows=[];os.makedirs('frozen-results',exist_ok=True)
# lock threshold .35 with .10 margin, exploratory only: inspired by calibrated dev score distributions, not tested operating guarantee.
for pair in range(12):
 ids=rng.choice(len(spks),3,replace=False);signals=[];centroids=[];paths=[]
 for idx in ids:
  fs=sorted(glob.glob(spks[idx]+'/*/*.flac'));valid=[f for f in fs if len(aud(f))>=sr*3]
  if len(valid)<4:valid=fs
  # Last long utterance as mixture input, first three distinct enrollment phrases.
  f=next(f for f in reversed(fs) if len(aud(f))>=n);en=[v for v in valid if v!=f][:3];e=np.mean([emb(aud(v)[:sr*3]) for v in en],axis=0);e=e/(np.linalg.norm(e)+1e-8);centroids.append(e);paths.append(dict(source=f,enrollment=en));x=aud(f)[-n:];signals.append(x/(np.sqrt(np.mean(x*x))+1e-8)*.06)
 for case in ['clean','noise10','absent','targetonly','otheronly','silence']:
  a,b,c=signals;mix=a+b
  if case=='noise10':mix=mix+rng.normal(size=n).astype(np.float32)*np.sqrt(np.mean(mix*mix)/10)
  if case=='absent':mix=b+c
  if case=='targetonly':mix=a
  if case=='otheronly':mix=b
  if case=='silence':mix=np.zeros(n,dtype=np.float32)
  outputs=[];cosines=[];accept=[];compute=[]
  for start in [0,32000]:
   ts=time.perf_counter();sources=sep.run(None,{'mixture':mix[start:start+32000].reshape(1,-1)})[0][0];rms=np.sqrt(np.mean(sources*sources,axis=1));es=[emb(s) if r>1e-5 else np.zeros_like(centroids[0]) for s,r in zip(sources,rms)];cos=np.array([np.dot(centroids[0],e) for e in es]);idx=int(np.argmax(cos));present=float(cos[idx])>.35 and float(cos[idx]-cos[1-idx])>.10 and rms[idx]>1e-5
   outputs.append(sources[idx]);cosines.append(cos.tolist());accept.append(bool(present));compute.append(time.perf_counter()-ts)
  y=np.concatenate(outputs);row=dict(pair=pair,case=case,paths=paths,cosines=cosines,accept=accept,improvement=sdr(y,a)-sdr(mix,a) if case in ['clean','noise10'] else None,selected_si_sdr=sdr(y,a) if case not in ['absent','otheronly','silence'] else None,rtf=sum(compute)/4);rows.append(row)
 print('pair',pair,flush=True)
json.dump(rows,open('frozen-results/metrics.json','w'),indent=2)
for case in ['clean','noise10','absent','targetonly','otheronly','silence']:
 rr=[r for r in rows if r['case']==case];print(case,'accept blocks',sum(sum(r['accept']) for r in rr),'/',len(rr)*2,'SI-SDRi',np.mean([r['improvement'] for r in rr]) if case in ['clean','noise10'] else '-', 'RTF',np.mean([r['rtf'] for r in rr]),flush=True)
