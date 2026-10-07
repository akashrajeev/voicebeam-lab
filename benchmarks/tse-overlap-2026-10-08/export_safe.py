import torch,onnxruntime as ort,numpy as np,time,json,soundfile as sf
from asteroid.models import ConvTasNet
torch.set_num_threads(2)
class SafeSep(torch.nn.Module):
 def __init__(self):
  super().__init__();self.m=ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k').eval()
 def forward(self,x):
  y=self.m(x);y=y/torch.sqrt(torch.mean(y*y,dim=-1,keepdim=True)+1e-8)
  a=y[:,0,:];b=y[:,1,:];v=x
  aa=torch.mean(a*a,dim=-1);bb=torch.mean(b*b,dim=-1);ab=torch.mean(a*b,dim=-1);av=torch.mean(a*v,dim=-1);bv=torch.mean(b*v,dim=-1)
  det=aa*bb-ab*ab
  ga=(av*bb-bv*ab)/(det+1e-6);gb=(bv*aa-av*ab)/(det+1e-6)
  return y*torch.stack([ga,gb],dim=-1).unsqueeze(-1)
m=SafeSep().eval();x=torch.randn(1,32000)*.08
with torch.inference_mode():torch.onnx.export(m,x,'convtasnet-2s-safe.onnx',input_names=['mixture'],output_names=['sources'],opset_version=17,dynamo=False)
s=ort.SessionOptions();s.intra_op_num_threads=2;s.inter_op_num_threads=1;r=ort.InferenceSession('convtasnet-2s-safe.onnx',s,providers=['CPUExecutionProvider']);rows=[]
for pair in range(6):
 a=sf.read(f'results-openss/pair{pair}-mix.wav',dtype='float32')[0][:32000].reshape(1,-1);times=[]
 for i in range(3):
  t=time.perf_counter();y=r.run(None,{'mixture':a})[0];times.append(time.perf_counter()-t)
 with torch.inference_mode():p=m(torch.tensor(a)).numpy()
 rows.append(dict(pair=pair,max_error=float(np.max(np.abs(y-p))),relative_rms=float(np.sqrt(np.mean((y-p)**2)/np.mean(p*p))),times=times,rtf=times[-1]/2,peak=float(np.max(np.abs(y))),reconstruction_rmse=float(np.sqrt(np.mean((np.sum(y,axis=1)-a)**2)))))
print(json.dumps(rows,indent=2));json.dump(rows,open('onnx-safe-parity.json','w'),indent=2)
