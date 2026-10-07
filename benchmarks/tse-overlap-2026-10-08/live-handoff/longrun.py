import onnxruntime as ort,numpy as np,soundfile as sf,time,resource,json
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;s=ort.InferenceSession('convtasnet-2s-safe.onnx',opts,providers=['CPUExecutionProvider']);x=sf.read('results-openss/pair2-mix.wav',dtype='float32')[0][:32000].reshape(1,-1);ts=[]
for i in range(100):
 t=time.perf_counter();y=s.run(None,{'mixture':x})[0];ts.append(time.perf_counter()-t)
print(json.dumps(dict(blocks=len(ts),seconds=float(sum(ts)),p50=float(np.percentile(ts,50)),p95=float(np.percentile(ts,95)),maximum=max(ts),maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finite=bool(np.isfinite(y).all()))))
