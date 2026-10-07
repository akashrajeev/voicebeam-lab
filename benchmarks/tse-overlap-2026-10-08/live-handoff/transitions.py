import json,time,glob,os
import numpy as np,soundfile as sf,sherpa_onnx,onnxruntime as ort
sr=16000;n=sr*9;context=16000;hop=4000;cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='titanet.onnx',num_threads=2,provider='cpu');enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg);o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;sep=ort.InferenceSession('convtasnet-1s-safe.onnx',o,providers=['CPUExecutionProvider']);rows=[];rng=np.random.default_rng(14008);os.makedirs('transition-results',exist_ok=True)
def emb(x):
 s=enc.create_stream();s.accept_waveform(sr,x);s.input_finished();e=np.array(enc.compute(s));return e/(np.linalg.norm(e)+1e-8)
def aud(f):return sf.read(f,dtype='float32')[0]
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
meta=json.load(open('frozen-results/metrics.json'))
for pair in range(2):
 r=next(r for r in meta if r['pair']==pair and r['case']=='clean');paths=r['paths'];signals=[]
 for p in paths:
  fs=sorted(glob.glob(os.path.dirname(p['source'])+'/*.flac'));f=next((f for f in reversed(fs) if len(aud(f))>=n),p['source']);x=aud(f);x=np.tile(x,1+n//len(x))[:n];signals.append(x/(np.sqrt(np.mean(x*x))+1e-8)*.06)
 a,b,c=signals;e=np.mean([emb(aud(p)[:sr*3]) for p in paths[0]['enrollment']],axis=0);e/=np.linalg.norm(e)
 for case in ['transition','noise10','targetonly','continuous']:
  target=a.copy();mix=a+b
  if case=='transition':target[3*sr:6*sr]=0;mix=target+b;mix[3*sr:6*sr]+=c[3*sr:6*sr]
  if case=='noise10':mix+=rng.normal(size=n).astype(np.float32)*np.sqrt(np.mean(mix*mix)/10)
  if case=='targetonly':mix=a.copy()
  out=np.zeros(n,dtype=np.float32);raw=np.zeros(n,dtype=np.float32);prev=None;blocks=[]
  for start in range(0,n,hop):
   end=start+hop;ws=end-context;lo=max(ws,0);inp=np.zeros(context,dtype=np.float32);inp[lo-ws:]=mix[lo:end];ts=time.perf_counter();sources=sep.run(None,{'mixture':inp.reshape(1,-1)})[0][0];es=[emb(v) for v in sources];cos=np.array([np.dot(e,v) for v in es]);idx=int(np.argmax(cos));freshidx=idx
   if prev is not None and abs(cos[0]-cos[1])<.10:
    overlap=context-hop;corr=[np.dot(prev[hop:],s[:overlap])/(np.linalg.norm(prev[hop:])*np.linalg.norm(s[:overlap])+1e-8) for s in sources];idx=int(np.argmax(corr))
   selected=sources[idx];seg=selected[-hop:].copy();present=bool(cos[idx]>.35 and cos[idx]-cos[1-idx]>.10);raw[start:end]=seg
   if not present:seg*=0
   if start>0:
    w=np.linspace(0,1,80);seg[:80]=out[start-1]*(1-w)+seg[:80]*w
   out[start:end]=seg;prev=selected.copy();blocks.append(dict(start=start/sr,cos=cos.tolist(),idx=idx,freshidx=freshidx,accepted=present,compute=time.perf_counter()-ts))
  row=dict(pair=pair,case=case,blocks=blocks,raw_improvement=sdr(raw[sr:],target[sr:])-sdr(mix[sr:],target[sr:]) if case in ['noise10','continuous'] else None,accepted_present=sum(b['accepted'] for b in blocks if (case!='transition' or b['start']<3 or b['start']>=6)),accepted_absent=sum(b['accepted'] for b in blocks if case=='transition' and 3<=b['start']<6));rows.append(row)
  if pair<2:sf.write(f'transition-results/p{pair}-{case}-mix.wav',mix,sr);sf.write(f'transition-results/p{pair}-{case}-out.wav',out,sr);sf.write(f'transition-results/p{pair}-{case}-raw.wav',raw,sr);sf.write(f'transition-results/p{pair}-{case}-target.wav',target,sr)
 print('pair',pair,flush=True)
json.dump(rows,open('transition-results/metrics.json','w'),indent=2)
for case in ['transition','noise10','targetonly','continuous']:
 rr=[r for r in rows if r['case']==case];print(case,'raw_SI-SDRi',np.mean([r['raw_improvement'] for r in rr]) if case in ['noise10','continuous'] else '-', 'accepted_present',sum(r['accepted_present'] for r in rr),'absent',sum(r['accepted_absent'] for r in rr));print('compute p50/p95',np.percentile([b['compute'] for r in rr for b in r['blocks']],[50,95]))
