import json,time,numpy as np,soundfile as sf,sherpa_onnx,onnxruntime as ort
sr=16000;n=sr*4;cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='titanet.onnx',num_threads=2,provider='cpu');enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg);opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;sep=ort.InferenceSession('convtasnet-2s-safe.onnx',opts,providers=['CPUExecutionProvider'])
def emb(x):
 s=enc.create_stream();s.accept_waveform(sr,x);s.input_finished();e=np.array(enc.compute(s));return e/(np.linalg.norm(e)+1e-8)
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
meta=json.load(open('frozen-results/metrics.json'));rows=[]
for pair in range(12):
 r=next(r for r in meta if r['pair']==pair and r['case']=='clean');pa=r['paths'][0];pb=r['paths'][1]
 a=sf.read(pa['source'],dtype='float32')[0][-n:];b=sf.read(pb['source'],dtype='float32')[0][-n:];a=a/(np.sqrt(np.mean(a*a))+1e-8)*.06;b=b/(np.sqrt(np.mean(b*b))+1e-8)*.06;mix=a+b;e=np.mean([emb(sf.read(p,dtype='float32')[0][:sr*3]) for p in pa['enrollment']],axis=0);e=e/np.linalg.norm(e)
 for mode in ['independent2s','ola1s','center1s']:
  outputs=np.zeros(n,dtype=np.float32);weights=np.zeros(n,dtype=np.float32);total=0;count=0
  starts=[0,32000] if mode=='independent2s' else ([-16000,0,16000,32000,48000] if mode=='ola1s' else [-8000,8000,24000,40000])
  for start in starts:
   lo=max(start,0);hi=min(start+32000,n);inp=np.zeros(32000,dtype=np.float32);inp[lo-start:hi-start]=mix[lo:hi];ts=time.perf_counter();sources=sep.run(None,{'mixture':inp.reshape(1,-1)})[0][0];es=[emb(s) for s in sources];idx=np.argmax([np.dot(e,v) for v in es]);y=sources[idx];total+=time.perf_counter()-ts;count+=1
   if mode=='independent2s':win=np.ones(32000,dtype=np.float32)
   elif mode=='ola1s':win=np.hanning(32000).astype(np.float32)
   else:win=np.zeros(32000,dtype=np.float32);win[8000:24000]=1
   outputs[lo:hi]+=y[lo-start:hi-start]*win[lo-start:hi-start];weights[lo:hi]+=win[lo-start:hi-start]
  outputs/=np.maximum(weights,1e-8);rows.append(dict(pair=pair,mode=mode,improvement=sdr(outputs,a)-sdr(mix,a),rtf=total/4,count=count,max_join=float(max(abs(outputs[i]-outputs[i-1]) for i in [16000,32000,48000]))));
json.dump(rows,open('continuity-metrics.json','w'),indent=2)
for mode in ['independent2s','ola1s','center1s']:
 rr=[r for r in rows if r['mode']==mode];print(mode,'mean',np.mean([r['improvement'] for r in rr]),'median',np.median([r['improvement'] for r in rr]),'RTF',np.mean([r['rtf'] for r in rr]),'degrade',sum(r['improvement']<0 for r in rr),'meanjoin',np.mean([r['max_join'] for r in rr]))
