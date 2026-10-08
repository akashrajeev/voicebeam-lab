import numpy as np,onnxruntime as ort,soundfile as sf,sherpa_onnx,json,time
from scipy.signal import lfilter
root='/home/sandbox/voicefix/skim-host';assets='/home/sandbox/voicefix/tse-repo/app/src/main/assets/selftest/'
def load(f):return sf.read(assets+f,dtype='float32')[0]
def norm(x,n=192000):x=np.resize(x,n);return x*.06/np.sqrt(np.mean(x*x))
a=norm(load('1089-134686-0002.wav'));b=norm(load('1221-135767-0005.wav'))
taps=np.array([-.00365145,0,.01617925,0,-.06841178,0,.30494752,.50187293,.30494752,0,-.06841178,0,.01617925,0,-.00365145],np.float32)
cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='/home/sandbox/voicefix/sep-host/titanet.onnx',num_threads=1,provider='cpu');enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
def emb(x):
 st=enc.create_stream();st.accept_waveform(16000,x);st.input_finished()
 if not enc.is_ready(st):return None
 e=np.asarray(enc.compute(st));return e/(np.linalg.norm(e)+1e-8)
e=emb(load('1089-134686-0013.wav')[:48000])
def up(x):
 y=np.zeros(len(x)*2,np.float32);y[1::2]=x;y[::2]=(np.r_[0,x[:-1]]+x)*.5;return y

def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
ss=ort.InferenceSession(root+'/skim-chunk-int8.onnx',opts,providers=['CPUExecutionProvider']);names={i.name for i in ss.get_inputs()};rng=np.random.default_rng(31)
rows=[]
for name,x in [('overlap',a+b),('target_only',a),('other_only',b),('silence',np.zeros_like(a)),('target_enters',b+np.r_[np.zeros(32000),a[32000:]]),('noisy_overlap',a+b+rng.normal(0,.02,len(a)).astype('float32'))]:
 x=x.astype('float32');down=lfilter(taps,[1.],x).astype('float32')[1::2];out=np.zeros((2,len(down)),np.float32);states=[np.zeros((1,1,384),np.float32) for _ in range(14)];tt=[];gain=None;tail=np.zeros((2,4),np.float32)
 for i in range(0,len(down)-204,200):
  feed={'audio':down[i:i+204][None,:],'active':np.array(float(i>0),np.float32),**{f'state_{j}':v for j,v in enumerate(states)}};t=time.perf_counter();ys=ss.run(None,{k:v for k,v in feed.items() if k in names});tt.append((time.perf_counter()-t)*1000);sources=ys[0][0].copy();sources[:,:4]+=tail;tail=ys[0][0][:,200:204].copy();sources=sources[:,:200];summ=sources.sum(axis=0);measured=float(np.dot(summ,down[i:i+200])/(np.dot(summ,summ)+1e-12));gain=measured if gain is None or gain*measured<=0 else .8*gain+.2*measured;sources*=gain;peak=np.abs(sources).max();sources*=min(1.,.95/(peak+1e-12));out[:,i:i+200]=sources;states=ys[1:]
 out=np.array([up(y) for y in out]);delivered=x.copy();selected=0;false_selected=0;first=None;decisions=[]
 for end in range(16000,len(x)-16000,16000):
  scores=[float(np.dot(v,e)) if (v:=emb(y[end-16000:end])) is not None else -1 for y in out];best=np.argmax(scores);yes=scores[best]>.35 and scores[best]-scores[1-best]>.1
  # Scores available after entire1sec reference window, apply NEXT second as production.
  if yes:delivered[end:end+16000]=out[best,end:end+16000];selected+=1;first=first or end//16
  decisions.append(dict(end_ms=end//16,scores=scores,selected=bool(yes)))
 # FIR/interpolation shift measured as8samples16k; align before quality score.
 sl=slice(4000,-1000);ref=a[sl];aligned=out[:,4007:-993]
 vals=[sdr(y,ref) for y in aligned]
 row={'case':name,'sep_p50_ms':float(np.median(tt[3:])),'sep_p95_ms':float(np.percentile(tt[3:],95)),'oracle_aligned_sdr':max(vals),'oracle_sdr_i':max(vals)-sdr(x[sl],ref),'selected_seconds':selected,'first_selected_ms':first,'scores':decisions,'peak':float(np.abs(out).max()),'finite':bool(np.isfinite(out).all())};rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='scores'}),flush=True)
json.dump(rows,open(root+'/e2e-results.json','w'),indent=2)
