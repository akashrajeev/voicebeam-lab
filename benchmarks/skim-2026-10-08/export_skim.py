import torch,numpy as np,time,json,onnxruntime as ort
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.separator.skim_separator import SkiMSeparator
from pathlib import Path
torch.set_num_threads(2)
root=Path('/home/sandbox/voicefix/skim-host')
enc=ConvEncoder(channel=128,kernel_size=8,stride=4)
sep=SkiMSeparator(128,causal=True,num_spk=2,layer=3,unit=384,segment_size=50,mem_type='hc',seg_overlap=False)
dec=ConvDecoder(channel=128,kernel_size=8,stride=4)
ck=torch.load(root/'149epoch.pth',map_location='cpu',weights_only=True)
for name,m in [('encoder',enc),('separator',sep),('decoder',dec)]:m.load_state_dict({k[len(name)+1:]:v for k,v in ck.items() if k.startswith(name+'.')});m.eval()
class Chunk(torch.nn.Module):
 def __init__(self):super().__init__();self.enc=enc;self.sep=sep;self.dec=dec
 def forward(self,audio,active,*state):
  sg=[(state[i*2],state[i*2+1]) for i in range(3)];ms=[[(state[6+i*4],state[7+i*4]),(state[8+i*4],state[9+i*4])] for i in range(2)]
  newsg=[(torch.zeros_like(sg[0][0]),torch.zeros_like(sg[0][1]))];newms=[]
  for i in range(2):
   ss,mm=self.sep.skim.mem_lstms[i].forward_one_step(sg[i],ms[i]);newsg.append(tuple(v*active for v in ss));newms.append([tuple(v*active for v in pair) for pair in mm])
  feat=self.enc.forward_streaming(audio)
  o=feat
  for i in range(3):o,newsg[i]=self.sep.skim.seg_lstms[i](o,newsg[i])
  masks=self.sep.nonlinear(self.sep.skim.output_fc(o.transpose(1,2)).transpose(1,2)).view(1,50,128,2)
  y=[self.dec.convtrans1d((feat*masks[:,:,:,i]).transpose(1,2)).squeeze(1) for i in range(2)]
  return (torch.stack(y,dim=1),*[v for pair in newsg for v in pair],*[v for pair in newms for sub in pair for v in sub])
m=Chunk().eval();z=[torch.zeros(1,1,384) for _ in range(14)];x=torch.randn(1,204)*.03
with torch.no_grad():
 # Compare chunk output to ESPnet exact per-frame streaming, across boundaries.
 st=None;state=z;errs=[]
 for k in range(4):
  xx=torch.randn(1,204)*.03;ref=[]
  for frame in enc.streaming_frame(xx):
   f=enc.forward_streaming(frame);ys,st,_=sep.forward_streaming(f,st);ref.append(torch.stack([dec.forward_streaming(y) for y in ys],dim=1))
  yy=m(xx,torch.tensor(float(k>0)),*state);state=list(yy[1:]);rr=torch.zeros(1,2,204)
  for j,v in enumerate(ref):rr[:,:,j*4:j*4+8]+=v
  errs.append(float((rr-yy[0]).abs().max()))
 print('stream_frame_vs_chunk_maxerr',errs,flush=True)
 names=['audio','active']+[f'state_{i}' for i in range(14)]
 torch.onnx.export(m,(x,torch.tensor(0.),*z),str(root/'skim-chunk-fp32.onnx'),input_names=names,output_names=['sources']+[f'next_{i}' for i in range(14)],opset_version=17,dynamo=False)
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;ss=ort.InferenceSession(str(root/'skim-chunk-fp32.onnx'),opts,providers=['CPUExecutionProvider']);state=[a.numpy() for a in z];tt=[];err=[]
 for k in range(30):
  xx=np.random.randn(1,204).astype('float32')*.03;feed={'audio':xx,'active':np.array(float(k>0),np.float32),**{f'state_{i}':v for i,v in enumerate(state)}}
  feed={k:v for k,v in feed.items() if k in [i.name for i in ss.get_inputs()]};t=time.perf_counter();yy=ss.run(None,feed);tt.append((time.perf_counter()-t)*1000)
  pp=m(torch.from_numpy(xx),torch.tensor(float(k>0)),*[torch.from_numpy(v) for v in state]);err.append(float(np.max(np.abs(yy[0]-pp[0].numpy()))));state=yy[1:]
 print('onnx_parity_max',max(err),'p50',np.median(tt[3:]),'p95',np.percentile(tt[3:],95),flush=True)
 json.dump({'stream_chunk_error':errs,'onnx_parity_error':max(err),'p50_ms':float(np.median(tt[3:])),'p95_ms':float(np.percentile(tt[3:],95))},open(root/'export-results.json','w'),indent=2)
