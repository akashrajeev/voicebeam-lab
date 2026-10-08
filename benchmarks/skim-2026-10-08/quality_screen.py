exec(open('/home/sandbox/voicefix/skim-host/export_skim.py').read().split('with torch.no_grad():')[0])
import soundfile as sf
from scipy.signal import resample_poly
from onnxruntime.quantization import quantize_dynamic,QuantType
from onnxconverter_common import float16
import onnx
assets='/home/sandbox/voicefix/tse-repo/app/src/main/assets/selftest/'
def read(f):
 x=sf.read(assets+f,dtype='float32')[0];x=np.resize(x,96000);x*=.06/np.sqrt(np.mean(x*x));return resample_poly(x,1,2).astype('float32')
a=read('1089-134686-0002.wav');b=read('1221-135767-0005.wav')
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
quantize_dynamic(str(root/'skim-chunk-fp32.onnx'),str(root/'skim-chunk-int8.onnx'),weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=['MatMul','LSTM'])
onnx.save(float16.convert_float_to_float16(onnx.load(root/'skim-chunk-fp32.onnx'),keep_io_types=True),root/'skim-chunk-fp16.onnx')
rows=[]
for kind in ['fp32','int8','fp16']:
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
 try:ss=ort.InferenceSession(str(root/f'skim-chunk-{kind}.onnx'),opts,providers=['CPUExecutionProvider'])
 except Exception as e:print(kind,'ERROR',e,flush=True);continue
 names={i.name for i in ss.get_inputs()}
 for case,x in [('overlap',a+b),('target_only',a)]:
  states=[np.zeros((1,1,384),np.float32) for _ in range(14)];out=np.zeros((2,len(x)+204),np.float32);tt=[]
  for i in range(0,len(x)-204,200):
   w=x[i:i+204][None,:];feed={'audio':w,'active':np.array(float(i>0),np.float32),**{f'state_{j}':v for j,v in enumerate(states)}}
   t=time.perf_counter();yy=ss.run(None,{k:v for k,v in feed.items() if k in names});tt.append((time.perf_counter()-t)*1000);out[:,i:i+204]+=yy[0][0];states=yy[1:]
  length=(len(x)-204)//200*200
  vals=[sdr(y[400:length],a[400:length]) for y in out];best=max(vals)
  row={'kind':kind,'case':case,'oracle_absolute_sdr':best,'oracle_sdr_i':best-sdr(x[400:length],a[400:length]),'p50_ms':float(np.median(tt[3:])),'p95_ms':float(np.percentile(tt[3:],95)),'size':(root/f'skim-chunk-{kind}.onnx').stat().st_size};rows.append(row);print(json.dumps(row),flush=True)
  if kind=='fp32':
   with torch.no_grad():
    tx=torch.from_numpy(x[:length])[None,:];fe,_=enc(tx,torch.tensor([length]));ys,_,_=sep(fe,torch.tensor([fe.shape[1]]));off=np.stack([dec(y,torch.tensor([length]))[0][0].numpy() for y in ys]);err=float(np.max(np.abs(off[:,400:length-8]-out[:,400:length-8])));print('offline_streaming_max_error',case,err,flush=True)
json.dump(rows,open(root/'quality-results.json','w'),indent=2)
