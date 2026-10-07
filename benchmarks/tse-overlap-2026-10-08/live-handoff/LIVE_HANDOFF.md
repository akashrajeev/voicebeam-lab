# Live experiment handoff, Oct 8 2026 01:05 IST

## Verdict
GO for an isolated, default-off phone experiment with actual overlap separation. NO-GO for claiming reliable hearing-grade isolation before on-phone room tests. No stable branch changed, no Actions minutes, no billing, no private audio. Branch mutation handed to engineering through parent; not performed here.

## Recommend first phone probe
1s context, 250ms hop, no extra lookahead. Graph `convtasnet-1s-safe.onnx`, input `mixture` float32 [1,16000], output `sources` float32 [1,2,16000], 16kHz. Gzip provided to engineering, 20MB uncompressed. SHA256 d65d34e718755ebfd8242c8b976657f1f56a7036d26b0c92d13d91eef4de7b04.
Keep 16kHz ring of 16000 samples. After each 4000 new samples, run on latest ring, select target source using frozen enrollment centroid + current TitaNet source embeddings; weak-margin waveform continuity only ties same person's waveform across overlapping contexts, NEVER proves target presence. Emit only newest 4000 samples. Fade first 80 samples from prior output sample (5ms). Ring needs zero-pad initially; evaluate both startup and postwarm, not just best middle.
No guarantee .35/.10 cutoff makes a reliable voice lock. Do not lower stable gate thresholds. In experiment, log both raw selection and gated output with a visibly uncertain/muted state, and measure the actual tradeoff. Current short-window policy drops too much speech.

## Measured live tradeoff
Six 6s synthetic two-speaker mixtures of real public held-out LibriSpeech speech with separate enrollment, requested target A. 3x3s centroid. Not room recordings; not Indian-accent validation. Score after context warm-up, source output corrected in graph.
- 1s context / 250ms hop / 0 lookahead: mean +12.412dB SI-SDRi, worst +9.331, 0/6 output clips closer to other speaker, mean of host p95 latency ~474.8ms, utilization .833.
- 1s / 250ms / 100ms lookahead: +12.145dB, worst +7.871, ~569.6ms, utilization .803. No clear benefit.
- 1s / 500ms / 0: +9.532dB, worst +1.862, ~740.2ms, utilization .446. Not the preferred 1s mode.
- 2s / 500ms / 0: +13.868dB, worst +12.322, ~862.9ms, utilization .668. Quality fallback if phone runs slowly.
- 2s / 250ms / 0: +13.187dB, ~621.5ms nominal, utilization 1.362: NOT sustainable, actual latency grows. Do not quote nominal latency as achievable.
Lookahead 100/200ms did not materially improve 2s clips. 250ms hop amortized source re-embedding once/second with continuity remained utilization >1 for 2s context (2-pair screen).
Host is 2-vCPU Xeon 2.60GHz, 1.9GiB. ORT 2 threads. No mobile CPU equivalent asserted. Latency estimate hop+lookahead+compute is nominal per-block worst/oldest sample budget, excludes device capture, resampling, scheduling, AudioTrack/Bluetooth buffers. Live means incremental delayed output, not negligible delay.
1s ONNX parity on six audio inputs max absolute error 3.83e-5. 2s ORT-only 100-block run: p50 .2268s, p95 .3020s, max .3410s, max process RSS 304756KB. This is neither an Android memory report nor a leak/thermal test.

## Absence transitions are still a blocker
Two 9s fixtures: target+other 0-3s, target absent with two others 3-6s, target returns 6-9s. 1s/250ms raw separation works but verification scores are weak. Provisional cosine>.35 and margin>.10 accepts 11/48 target-present hops, 0/24 target-absent hops. Continuous overlap accepts 24/72, noise 0/72, target-only 44/72. Not usable listening coverage.
3s selected-voice reservoir improves continuous acceptance to 50/72 and target-only to 59/72, but target transitions accept only 14/48 present hops and falsely pass 4/24 absent hops. Reservoir lags absence and return, extra compute .24-.29s undermines 250ms budget. Do NOT claim reservoir fixed it. Target persistence must not mask presence uncertainty.
Need actual-device test with target enrolled then target absent, other voice alone, overlaps, target return, same-sex voice pair, turn changes. User confirmation of behavior and logs is a gate, not a formality.

## Independent buffered quality screen
2s blocks, sign/scale fixed, TitaNet selection, 20 new pairs/40 target requests: clean +15.16dB mean (0 degrade, 38 >=6), synthetic10dB noise +9.53 (1 degrade), synthetic reverb +6.02 (4 degrade), three-speaker +2.52 (12 degrade). These are separation-only outputs, not gated listening scores. model supports two speakers, not arbitrary crowds.

## Training/fine-tuning feasibility
No GPU is exposed here; available free Actions scope is CPU Linux runners, not a promised GPU allocation. No trained causal checkpoint with resolved provenance was recovered in the screened small-model options. Training a new causal model is not a credible finale path: need data generation, speaker conditioning, an actual optimizer run, validation across speakers, streaming-state export, mobile conversion and phone tuning. Minutes of fine-tuning can't establish quality; fine-tuning the noncausal ConvTasNet does not turn its future-looking dilations/global normalization into a causal model. No training started and no free-minute quota spent. Keep this as a later rebuild experiment, not a substitute for the demonstrated route.

## Timing and go/no-go gates
Phone test can begin once separate experiment APK integrates 1s graph and instruments p50/p95 separator+TitaNet, queue depth/drops, route latency, memory, battery/heat. Proposed engineering acceptance: p95 combined compute <=200ms at 250ms hop with sustained 5min run, zero backlog, finite output, no speaker switch, and correct target-absent behavior. These are proposed targets, not measured phone facts. If slower, widen hop or use 2s/500ms quality mode and disclose actual delay; never silently let queue grow.
Ideally test before proposed Oct8 14:00 pretravel freeze. Travel/device availability is not verified. Finale report Fri Oct9 17:00 is source-grounded in parent context; that is a hard boundary, not an APK-test promise. Ask Akash when he can test now rather than assume productive bus hours. Need repair buffer before arrival.

## Licenses and sources
- https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k : accessible pretrained 5,066,929-parameter two-speaker separator. Card training train-360, 16kHz. Code Asteroid MIT, weights metadata CC-BY-SA4.0 vs body CC-BY-SA3.0. Preserve attribution/share-alike and report version conflict; not a clean permissive redistribution claim.
- https://asteroid.readthedocs.io/en/v0.5.0/readmes/pretrained_models.html : says model releases use CC-BY-SA3.0 and NC training sources imply NC models. Selected clean LibriSpeech route avoids explicitly noisy NC model.
- https://www.openslr.org/12 : LibriSpeech CC-BY4.0, Vassil Panayotov et al., derived from LibriVox. Original individual fixture filenames retained in metrics.
- https://github.com/helloooideeeeea/OpenSpeakerBeam-SS : MIT cloned code, older fetched README TBD. 33.05MB checkpoint https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset has no license card/source split provenance. Private evaluated, excluded from shipping.
- https://github.com/wenet-e2e/wesep : current source says official pretrained packages coming soon, not proof a phone release is ready.
- https://github.com/Zachshotamartin/target-speaker-extraction : 2.26M custom model promising results but README explicitly says original code license not selected and weights protected. Excluded.
- https://github.com/k2-fsa/sherpa-onnx/issues/3674 : reports VoiceSplit separation unreliable on longer/target-silent clips, not a substitute for measurements.
- Exact TitaNet asset https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx ; SHA256 ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e.
