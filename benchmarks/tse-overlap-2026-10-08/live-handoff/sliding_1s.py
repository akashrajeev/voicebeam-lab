import json,time,sys,os,glob
import numpy as np,soundfile as sf,sherpa_onnx,onnxruntime as ort
sr=16000;duration=6;n=sr*duration;cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='titanet.onnx',num_threads=2,provider='cpu');enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg);opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;sep=ort.InferenceSession('convtasnet-1s-safe.onnx',opts,providers=['CPUExecutionProvider']);pair=int(sys.argv[1]);os.makedirs('sliding-1s',exist_ok=True)
def emb(x):
 s=enc.create_stream();s.accept_waveform(sr,x);s.input_finished();e=np.array(enc.compute(s));return e/(np.linalg.norm(e)+1e-8)
def aud(f):return sf.read(f,dtype='float32')[0]
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
meta=json.load(open('frozen-results/metrics.json'));r=next(r for r in meta if r['pair']==pair and r['case']=='clean');paths=r['paths'];signals=[]
for p in paths[:2]:
 fs=sorted(glob.glob(os.path.dirname(p['source'])+'/*.flac'));f=next(f for f in reversed(fs) if len(aud(f))>=n);x=aud(f)[-n:];x=x/(np.sqrt(np.mean(x*x))+1e-8)*.06;signals.append(x)
a,b=signals;mix=a+b;e=np.mean([emb(aud(p)[:sr*3]) for p in paths[0]['enrollment']],axis=0);e/=np.linalg.norm(e);rows=[]
for hop in [.5,.25]:
 for look in [0,.1]:
  h=int(sr*hop);d=int(sr*look);out=np.zeros(n,dtype=np.float32);times=[];sep_times=[];choices=[];sims=[];prev=None;switches=0
  for start in range(0,n,h):
   end=start+h+d;ws=end-16000;lo=max(ws,0);hi=min(end,n);inp=np.zeros(16000,dtype=np.float32);inp[lo-ws:hi-ws]=mix[lo:hi];t=time.perf_counter();sources=sep.run(None,{'mixture':inp.reshape(1,-1)})[0][0];st=time.perf_counter()-t
   # Align speaker labels by waveform correlation in overlapping context. Fresh target match wins unless margin is weak.
   es=[emb(s) for s in sources];cos=np.array([np.dot(e,v) for v in es]);idx=int(np.argmax(cos));
   if prev is not None and abs(cos[0]-cos[1])<.10:
    overlap=16000-h;corr=[np.dot(prev[h:],s[:overlap])/(np.linalg.norm(prev[h:])*np.linalg.norm(s[:overlap])+1e-8) for s in sources];idx=int(np.argmax(corr))
   selected=sources[idx];segment=selected[16000-d-h:16000-d]
   # Short causal seam fade: extrapolate last sample into new 5ms ramp, no future dependency.
   if start>0:
    fade=min(80,len(segment));w=np.linspace(0,1,fade);segment[:fade]=out[start-1]*(1-w)+segment[:fade]*w
   out[start:min(start+h,n)]=segment[:min(h,n-start)];prev=selected.copy();times.append(time.perf_counter()-t);sep_times.append(st);choices.append(idx);sims.append(cos.tolist())
  improvement=sdr(out[16000:],a[16000:])-sdr(mix[16000:],a[16000:]);row=dict(pair=pair,hop=hop,lookahead=look,improvement_postwarm=improvement,improvement_all=sdr(out,a)-sdr(mix,a),si_sdr=sdr(out[16000:],a[16000:]),other_si_sdr=sdr(out[16000:],b[16000:]),compute_p50=float(np.percentile(times,50)),compute_p95=float(np.percentile(times,95)),separator_p50=float(np.percentile(sep_times,50)),utilization=float(np.mean(times)/hop),latency_p50=hop+look+float(np.percentile(times,50)),latency_p95=hop+look+float(np.percentile(times,95)),choices=choices,cosines=sims);rows.append(row);print(json.dumps(row),flush=True);sf.write(f'sliding-1s/p{pair}-hop{hop}-look{look}.wav',out,sr)
sf.write(f'sliding-1s/p{pair}-mix.wav',mix,sr);sf.write(f'sliding-1s/p{pair}-ref.wav',a,sr);json.dump(rows,open(f'sliding-1s/p{pair}.json','w'),indent=2)
