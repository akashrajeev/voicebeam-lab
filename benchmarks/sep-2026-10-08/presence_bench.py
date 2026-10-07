import numpy as np,soundfile as sf,sherpa_onnx,onnxruntime as ort,json,time
root='/home/sandbox/voicefix/tse-repo/app/src/main/assets'
cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='/home/sandbox/voicefix/sep-host/titanet.onnx',num_threads=1,provider='cpu');enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
def emb(x):
 s=enc.create_stream();s.accept_waveform(16000,x);s.input_finished()
 if not enc.is_ready(s):return None
 e=np.asarray(enc.compute(s));return e/(np.linalg.norm(e)+1e-8)
def read(n):return sf.read(root+'/selftest/'+n,dtype='float32')[0]
e=emb(read('1089-134686-0013.wav')[:48000])
a=read('1089-134686-0002.wav');b=read('1221-135767-0005.wav')
def rep(x):return np.resize(x,64000)*(.06/np.sqrt(np.mean(x*x)))
a=rep(a);b=rep(b)
cases={'target_only':a,'other_only':b,'silence':np.zeros(64000,np.float32),'target_enters':b+np.concatenate([np.zeros(32000,np.float32),a[32000:]])}
rows=[]
for n in [16000,32000]:
 hop=n//4
 for kind in ['safe','int8']:
  opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
  sess=ort.InferenceSession(root+f'/models/convtasnet-{n//16000}s-{kind}.onnx',opt,providers=['CPUExecutionProvider'])
  for case,x in cases.items():
   chunks=[[],[]];prev=None;cached=[-1.,-1.];negatives=0;updated=-100000;selected=False;negative=False
   for end in range(hop,len(x)+1,hop):
    w=np.zeros(n,np.float32);lo=max(0,end-n);w[n-(end-lo):]=x[lo:end];src=sess.run(None,{'mixture':w[None,:]})[0][0];order=[0,1]
    if prev is not None:
     def corr(p,s):return float(np.dot(p[hop:],s[:-hop])/(np.linalg.norm(p[hop:])*np.linalg.norm(s[:-hop])+1e-8))
     if corr(prev[0],src[1])+corr(prev[1],src[0])>corr(prev[0],src[0])+corr(prev[1],src[1]):order=[1,0]
    prev=src[order].copy()
    for c in range(2):chunks[c].extend(src[order[c]][-hop:])
    fresh=len(chunks[0])>=16000
    if fresh:
     es=[emb(np.asarray(c[-16000:],np.float32)) for c in chunks];cached=[float(np.dot(z,e)) if z is not None else -1. for z in es];chunks=[[],[]]
    scores=[0.,0.]
    for c in range(2):scores[order[c]]=cached[c]
    valid=all(np.isfinite(s) and s>=0 for s in scores);now=end//16
    if fresh:
     updated=now;hi=max(scores);lo=min(scores);selected=valid and hi>.35 and hi-lo>.10;negative=valid and hi<.25;negatives=negatives+1 if negative else 0
    decision='SELECT' if valid and selected and now-updated<=1000 else 'MUTE' if valid and negative and negatives>=2 and now-updated<=250 else 'RAW'
    rows.append({'context_ms':n//16,'kind':kind,'case':case,'end_ms':now,'scores':scores,'fresh':fresh,'decision':decision})
json.dump(rows,open('/home/sandbox/voicefix/sep-host/presence-results.json','w'),indent=2)
for case in cases:
 rr=[r for r in rows if r['case']==case];print(case,{k:sum(r['decision']==k for r in rr) for k in ['SELECT','MUTE','RAW']},'last scores',[r['scores'] for r in rr if r['end_ms']==4000])
