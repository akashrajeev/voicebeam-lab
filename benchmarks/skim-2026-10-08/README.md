# SEP-9 pretrained causal SkiM, October8
Private experiment, no Pixel proof yet.8k WSJ0-2Mix two-speaker model; current phone audio16k. Weights31.5MB:
https://huggingface.co/lichenda/wsj0_2mix_skim_small_causal/resolve/main/exp/enh_train_enh_skim_causal_small_raw/149epoch.pth?download=true
Config: https://huggingface.co/lichenda/wsj0_2mix_skim_small_causal/resolve/main/exp/enh_train_enh_skim_causal_small_raw/config.yaml?download=true
Card declares CC-BY4.0. Preserve attribution; no public release in this work.
Export:25ms blocks204samples8k,200new samples+4decoder overlap,14state outputs, first-block active=0 then1. Memory boundary every50encoder frames. State_4/5 unused and pruned; output next_0..13 still present.3seg LSTMs,2memory(h/c)LSTMs. Batch1 only.
FP32 streaming-vs-ESPnet per-frame<4e-7; ONNX-vs-PyTorch<9e-7; valid offline-vs-streaming fixture max1.72e-5(overlap),5.72e-6(clean). Final incomplete overlap-add tail excluded.
Dynamic int8 LSTM/MatMul per-channel8.24MB; fp3231.55MB,fp1615.81MB. One public LibriSpeech pair,12sec end-to-end fixtures. HostCPU only, not Pixel predictions.
Android-style15tap antialias downsample + linear interpolation has7sample16k delay. Sources require polarity/gain normalization against their sum reconstruction BEFORE peak limiting; SI-SDR alone hides arbitrary model scale. Gain smoothed within same sign.
Latest end-to-end oracle aligned overlap SI-SDRi13.40dB, noisy11.94dB, clean absolute17.56dB. All finite, peak<.69, silence0. Correct8k processing costs audible bandwidth reduction.
TitaNet identity1Hz separate worker. Positive threshold.35/margin.1, never negative mute. Selection:target-only8/10seconds, overlap4/10,noisy2/10, other-only/silence0; target-entry3sec recovery. These source selection limitations remain. No claim of robust arbitrary chatter isolation.
200ms fixed software playback delay, sample-indexed delayed raw on missing/uncertain output. Warmup silence explicit. Late SEP never replayed; packet gaps reset all recurrent/resampler and identity state. This value is NOT acoustic/BT latency. DefaultSEP remainsOFF. ENH alignment/valid enrollment/selected-only safe boost kept, Bluetooth routing untouched.

# First Pixel matrix, only after fresh free quota check
Pixel8API34, one physical matrix.6cases:target-only,other-only,silence,overlap,target_enters2sec,noisy_overlap. Model25ms p50/p95/deadline misses, finite/peak, absolute reconstruction/SI-SDRi. Paced production6sec each:200ms delay, fallback/selected/warmup frames, first positive selection, delivered-policy SI-SDR; log chunks/inputdrops/late/backlog. Auto3x2sec reference scripted only. No mic/camera/AudioTrack/BT in this test. Additional real-room/camera/BT/heat/user listening remains necessary after it. Passed means execution not performance verdict.
