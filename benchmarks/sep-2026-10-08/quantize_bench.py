import os,json,time
import numpy as np,onnx,onnxruntime as ort,soundfile as sf
from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat,QuantType
from onnxconverter_common import float16
ROOT='/home/sandbox/voicefix/tse-repo/app/src/main/assets'
a,sr=sf.read(ROOT+'/selftest/1089-134686-0002.wav',dtype='float32');b,_=sf.read(ROOT+'/selftest/1221-135767-0005.wav',dtype='float32')
def rep(x):return np.resize(x,96000)*(.06/np.sqrt(np.mean(x*x)))
a=rep(a);b=rep(b);mix=a+b
class Reader(CalibrationDataReader):
 def __init__(self,n):self.it=iter([{'mixture':mix[i:i+n][None,:]} for i in range(0,96000-n+1,4000)])
 def get_next(self):return next(self.it,None)
def sdr(y,x):
 y=y-y.mean();x=x-x.mean();p=np.dot(y,x)*x/(np.dot(x,x)+1e-12);return float(10*np.log10((np.dot(p,p)+1e-12)/(np.dot(y-p,y-p)+1e-12)))
rows=[]
for n in [16000,32000]:
 src=f'{ROOT}/models/convtasnet-{n//16000}s-safe.onnx'
 m=onnx.load(src);name=m.graph.input[0].name
 assert name=='mixture',name
 q=f'/home/sandbox/voicefix/sep-host/convtasnet-{n//16000}s-int8.onnx'
 try:
  quantize_static(src,q,Reader(n),quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=['Conv'])
 except Exception as e:print('quantize failure',str(e),flush=True);continue
 f=f'/home/sandbox/voicefix/sep-host/convtasnet-{n//16000}s-fp16.onnx'
 onnx.save(float16.convert_float_to_float16(m,keep_io_types=True,disable_shape_infer=True),f)
 for kind,path in [('fp32',src),('int8',q),('fp16',f)]:
  try:
   opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
   sess=ort.InferenceSession(path,opt,providers=['CPUExecutionProvider']);times=[];imp=[]
   for i in range(0,96000-n+1,8000):
    x=mix[i:i+n][None,:];t=time.perf_counter();y=sess.run(None,{name:x})[0][0];times.append((time.perf_counter()-t)*1000)
    imp.append(max(sdr(z,a[i:i+n]) for z in y)-sdr(x[0],a[i:i+n]))
   row={'context':n,'kind':kind,'size':os.path.getsize(path),'p50_ms':float(np.median(times[1:])),'p95_ms':float(np.percentile(times[1:],95)),'mean_oracle_sdr_i':float(np.mean(imp))};rows.append(row);print(json.dumps(row),flush=True)
  except Exception as e:row={'context':n,'kind':kind,'error':str(e)};rows.append(row);print(json.dumps(row),flush=True)
json.dump(rows,open('/home/sandbox/voicefix/sep-host/results.json','w'),indent=2)
