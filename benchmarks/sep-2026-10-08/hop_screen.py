import numpy as np, soundfile as sf, onnxruntime as ort, time,json
root='/home/sandbox/voicefix/tse-repo/app/src/main/assets'
def load(f):
 x=sf.read(root+'/selftest/'+f,dtype='float32')[0];return np.resize(x,192000)*(.06/np.sqrt(np.mean(x*x)))
a=load('1089-134686-0002.wav');b=load('1221-135767-0005.wav');mix=a+b
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
rows=[]
for n in [16000,32000]:
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 ss=ort.InferenceSession(root+f'/models/convtasnet-{n//16000}s-int8.onnx',opt,providers=['CPUExecutionProvider'])
 for hop in [4000,8000,16000]:
  if hop>n:continue
  for case,m in [('overlap',mix),('target_only',a)]:
   out=np.zeros_like(a);tt=[];prev=None;switch=0
   for end in range(hop,len(a)+1,hop):
    w=np.zeros(n,np.float32);lo=max(0,end-n);w[n-end+lo:]=m[lo:end];t=time.perf_counter();src=ss.run(None,{'mixture':w[None,:]})[0][0];tt.append((time.perf_counter()-t)*1000)
    # Oracle source score isolates hop-boundary quality, NOT identity/policy.
    ref=np.zeros(n,np.float32);ref[n-end+lo:]=a[lo:end]
    k=np.argmax([sdr(z,ref) for z in src]);out[end-hop:end]=src[k][-hop:]
   rows.append(dict(context_ms=n//16,hop_ms=hop//16,case=case,sep_p50_ms=float(np.median(tt[1:])),sep_p95_ms=float(np.percentile(tt[1:],95)),oracle_absolute_sdr=sdr(out[n:],a[n:]),oracle_sdr_i=sdr(out[n:],a[n:])-sdr(m[n:],a[n:]),projected_sep_misses=sum(t>hop/16 for t in tt),hops=len(tt)))
   print(json.dumps(rows[-1]),flush=True)
json.dump(rows,open('/home/sandbox/voicefix/sep-host/hop-results.json','w'),indent=2)
