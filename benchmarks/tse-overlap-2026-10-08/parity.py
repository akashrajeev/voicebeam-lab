import torch,onnxruntime as ort,numpy as np,soundfile as sf,json
from asteroid.models import ConvTasNet
torch.set_num_threads(2);m=ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k').eval();cfg=ort.SessionOptions();cfg.intra_op_num_threads=2;cfg.inter_op_num_threads=1;s=ort.InferenceSession('convtasnet-2s.onnx',cfg,providers=['CPUExecutionProvider']);rr=[]
for pair in range(6):
 x=sf.read(f'results-openss/pair{pair}-mix.wav',dtype='float32')[0][:32000].reshape(1,-1)
 with torch.inference_mode():a=m(torch.tensor(x)).numpy()
 b=s.run(None,{'mixture':x})[0];d=a-b;rr.append(dict(pair=pair,max_abs=float(np.max(np.abs(d))),rms_error=float(np.sqrt(np.mean(d*d))),relative_rms=float(np.sqrt(np.mean(d*d)/np.mean(a*a))),peak=float(np.max(np.abs(a)))))
print(json.dumps(rr,indent=2));json.dump(rr,open('onnx-parity.json','w'),indent=2)
