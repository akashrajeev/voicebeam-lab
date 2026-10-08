import sherpa_onnx as s,onnxruntime as ort,soundfile as sf,numpy as np,json,time
root='/home/sandbox/voicefix/tse-repo/app/src/main/assets';a,sr=sf.read(root+'/selftest/1089-134686-0002.wav',dtype='float32');b,_=sf.read(root+'/selftest/1221-135767-0005.wav',dtype='float32')
def rep(x):return np.resize(x,64000)*(.06/np.sqrt(np.mean(x*x)))
a=rep(a);b=rep(b);rng=np.random.default_rng(41)
def den(x):
 c=s.OnlineSpeechDenoiserConfig(model=s.OfflineSpeechDenoiserModelConfig(gtcrn=s.OfflineSpeechDenoiserGtcrnModelConfig(model='/home/sandbox/voicefix/sep-host/gtcrn.onnx'),num_threads=1));d=s.OnlineSpeechDenoiser(c);out=[]
 for i in range(0,len(x),256):out.extend(d.run(x[i:i+256],16000).samples)
 out.extend(d.flush().samples);return np.array(out,np.float32)[:len(x)]
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
rows=[]
for n in [16000,32000]:
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 sess=ort.InferenceSession(root+f'/models/convtasnet-{n//16000}s-int8.onnx',opt,providers=['CPUExecutionProvider'])
 for noise in [0,.01,.025]:
  mix=a+b+rng.normal(0,noise,a.size).astype(np.float32)
  for stage in ['none','pre','post']:
   x=den(mix) if stage=='pre' else mix;y=[];ts=time.perf_counter()
   for pos in range(0,len(x),n):
    sources=sess.run(None,{'mixture':x[pos:pos+n][None,:]})[0][0];best=max(sources,key=lambda z:sdr(z,a[pos:pos+n]));y.extend(best)
   y=np.array(y,np.float32);y=den(y) if stage=='post' else y
   # Search alignment, no naive pointwise dry/wet mixing.
   opts=[(lag,sdr(y[max(lag,0):len(y)+min(lag,0)],a[max(-lag,0):len(a)-max(lag,0)])) for lag in range(-512,513,16)]
   lag,score=max(opts,key=lambda z:z[1]);row={'context':n,'noise_rms':noise,'stage':stage,'oracle_sdr_i':score-sdr(mix,a),'best_lag_samples':lag,'elapsed_ms':(time.perf_counter()-ts)*1000};rows.append(row);print(json.dumps(row),flush=True)
json.dump(rows,open('/home/sandbox/voicefix/sep-host/denoise-results.json','w'),indent=2)
