import glob,os,time,json
import numpy as np,soundfile as sf,sherpa_onnx,onnxruntime as ort
from scipy.signal import fftconvolve
sr=16000;n=sr*4;cfg=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model='titanet.onnx',num_threads=2,provider='cpu');enc=sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;sep=ort.InferenceSession('convtasnet-2s.onnx',opts,providers=['CPUExecutionProvider']);rng=np.random.default_rng(81108)
def emb(x):
 s=enc.create_stream();s.accept_waveform(sr,x);s.input_finished();e=np.array(enc.compute(s));return e/(np.linalg.norm(e)+1e-8)
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
def aud(f):return sf.read(f,dtype='float32')[0]
def choose(spk,offset=0):
 fs=sorted(glob.glob(spk+'/*/*.flac'));idx=next(i for i,f in enumerate(fs[offset:]) if len(aud(f))>=n)+offset;return fs[idx],fs[(idx+1)%len(fs)]
spks=sorted(glob.glob('fixtures/LibriSpeech/test-clean/*'));rows=[];os.makedirs('stress-scaled',exist_ok=True)
# Deterministic 20 held-out pair configurations, no parameter tuning on results.
for pair in range(20):
 ia=(pair*7)%len(spks);ib=(ia+13)%len(spks);ic=(ia+23)%len(spks)
 fa,ea=choose(spks[ia],3);fb,eb=choose(spks[ib],3);fc,ec=choose(spks[ic],3)
 x=aud(fa)[:n];z=aud(fb)[:n];third=aud(fc)[:n];x=x/(np.sqrt(np.mean(x*x))+1e-8)*.06;z=z/(np.sqrt(np.mean(z*z))+1e-8)*.06;third=third/(np.sqrt(np.mean(third*third))+1e-8)*.06
 es=[emb(aud(e)[:sr*9]) for e in [ea,eb,ec]]
 for case in ['clean','noise10','reverb','three','absent']:
  # Simple synthetic reverberation, not a recorded room impulse response.
  refs=[x,z];mix=x+z
  if case=='noise10':mix=mix+rng.normal(size=n).astype(np.float32)*np.sqrt(np.mean(mix*mix)/10)
  if case=='reverb':
   ir=np.zeros(sr//2,dtype=np.float32);ir[0]=1
   for delay,gain in [(.043,.4),(.079,.3),(.127,.2),(.233,.15)]:ir[int(delay*sr)]=gain
   refs=[fftconvolve(v,ir)[:n].astype(np.float32) for v in refs];mix=refs[0]+refs[1]
  if case=='three':mix=mix+third
  if case=='absent':mix=z+third
  outs=[[],[]];scores=[[],[]];oracle=[[],[]];latencies=[];choices=[[],[]]
  for start in [0,32000]:
   inp=mix[start:start+32000];ts=time.perf_counter();o=sep.run(None,{'mixture':inp.reshape(1,-1)})[0][0];g=np.linalg.lstsq(o.T,inp,rcond=1e-6)[0];o=o*g[:,None];ee=[emb(v) for v in o];latencies.append(time.perf_counter()-ts)
   for target in [0,1]:
    cos=[float(np.dot(es[target],e)) for e in ee];idx=int(np.argmax(cos));outs[target].append(o[idx]);scores[target].append(cos);choices[target].append(idx)
    oracle[target].append(o[np.argmax([sdr(v,refs[target][start:start+32000]) for v in o])])
  for target in [0,1]:
   y=np.concatenate(outs[target]);row=dict(pair=pair,case=case,target=target,source_a=fa,source_b=fb,source_c=fc,enroll=[ea,eb,ec][target],improvement=None if case=='absent' and target==0 else sdr(y,refs[target])-sdr(mix,refs[target]),si_sdr=sdr(y,refs[target]),other_si_sdr=sdr(y,refs[1-target]),oracle_improvement=sdr(np.concatenate(oracle[target]),refs[target])-sdr(mix,refs[target]),cosines=scores[target],choices=choices[target],rtf=sum(latencies)/4,block_seconds=latencies);rows.append(row)
   if pair<2 or (case=='clean' and pair==5):sf.write(f'stress-scaled/p{pair}-{case}-target{target}.wav',y,sr)
  if pair<2:sf.write(f'stress-scaled/p{pair}-{case}-mix.wav',mix,sr);sf.write(f'stress-scaled/p{pair}-{case}-ref.wav',refs[0],sr)
 print('pair done',pair,flush=True)
json.dump(rows,open('stress-scaled/metrics.json','w'),indent=2)
for case in ['clean','noise10','reverb','three']:
 rr=[r for r in rows if r['case']==case];vals=[r['improvement'] for r in rr];print(case,dict(mean=float(np.mean(vals)),median=float(np.median(vals)),minimum=min(vals),degraded=sum(v<0 for v in vals),above6=sum(v>=6 for v in vals),count=len(vals),rtf=float(np.mean([r['rtf'] for r in rr]))),flush=True)
