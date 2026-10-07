import torch,onnxruntime as ort,numpy as np,time,json
from asteroid.models import ConvTasNet
torch.set_num_threads(2)
m=ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k').eval();x=torch.randn(1,32000)
with torch.inference_mode():
 torch.onnx.export(m,x,'convtasnet-2s.onnx',input_names=['mixture'],output_names=['sources'],opset_version=17,dynamo=False)
s=ort.SessionOptions();s.intra_op_num_threads=2;s.inter_op_num_threads=1
r=ort.InferenceSession('convtasnet-2s.onnx',s,providers=['CPUExecutionProvider']);a=x.numpy();times=[]
for i in range(10):
 t=time.perf_counter();y=r.run(None,{'mixture':a})[0];times.append(time.perf_counter()-t)
with torch.inference_mode(): p=m(x).numpy()
print(json.dumps({'max_error':float(np.max(np.abs(y-p))),'times':times,'mean_rtf':float(np.mean(times[1:])/2)}))
