# VoiceBeam overlap experiment, Oct 8 2026, 00:45 IST

Use a separate branch. Stable Bluetooth/gate unchanged. Private public-audio evaluation only so far. No Actions minutes, billing, or private audio used.

## Concrete route
JorisCos/ConvTasNet_Libri2Mix_sepclean_16k, downloaded from Hugging Face via Asteroid. Two-speaker blind separation, then existing TitaNet selects the enrolled voice. This is real overlap separation, not target conditioning inside the separator. Fixed 2-second windows introduce at least 2 seconds of buffering. Hearing-earpiece real-time behavior is not validated.

IMPORTANT: outputs have arbitrary huge scaling/signs. Use `g = np.linalg.lstsq(sources.T, mixture, rcond=1e-6)[0]; sources *= g[:,None]` before embedding or playing PCM. Tests before this correction produced clipping and wrong speaker picks. The corrected stress-scaled results are authoritative; earlier metrics are preliminary.

## Measurements
Host: 2-vCPU x86 Xeon 2.60 GHz, 1.9 GiB RAM, PyTorch/ORT CPU, 2 threads. Not a phone proxy strong enough to promise Android performance.
20 deterministic new 4-second pairs, 2 requests each, equal-RMS real LibriSpeech test-clean utterances, separate same-speaker enrollment; no tuned thresholds.
2s independent ONNX windows + TitaNet + least-squares rescale:
- Clean: mean SI-SDR improvement +15.164 dB, median +16.029, min +2.425; 0/40 degrade; 38/40 >=6dB.
- Synthetic 10dB Gaussian noise: +9.530 mean; 1/40 degrade; 37/40 >=6dB.
- Synthetic 4-echo reverb (not room RIR): +6.017 mean; 4/40 degrade; 28/40 >=6dB.
- Three voices: +2.518 mean; 12/40 degrade; 9/40 >=6dB. Model is trained for TWO speakers.
- Target absent: 0/40 blocks exceed cosine .53. Same cutoff accepts only 54/80 clean target-present blocks, 7/80 noise, 8/80 reverb. Threshold not calibrated. Mute/uncertain state needed.
- Mean host RTF ~.15 for ONNX separation, rescale and both TitaNet embeddings. Phone latency, thermals, allocation, routes untested.
- ONNX exported fixed [1,32000] mixture -> [1,2,32000] sources; 20MB. Relative RMS parity error vs PyTorch ~3.2e-5 to 6.2e-5 on six audio inputs. Absolute error alone misleading due to enormous unscaled outputs.

## Sources and license caveats
Model https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k : card metadata CC-BY-SA 4.0, body CC-BY-SA 3.0. Attribution/share-alike must be preserved; redistribution legal/version ambiguity remains, not a permissive MIT model. Train split in card is LibriSpeech train-clean-360, validation dev. test-clean source utterances are held out from that declared training split, but these are not official Libri2Mix fixtures.
Asteroid https://asteroid.readthedocs.io/en/v0.5.0/readmes/pretrained_models.html : model license policy CC-BY-SA 3.0.
Audio https://www.openslr.org/12 : CC-BY 4.0, Vassil Panayotov et al., derived LibriVox audiobooks. Keep individual fixture source filenames in metrics for attribution.
TitaNet exact asset https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx ; received from engineering and downloaded live. Same extractor API as Android, no manual MFCC.

## Other candidate
OpenSpeakerBeam-SS https://github.com/helloooideeeeea/OpenSpeakerBeam-SS : cloned source MIT, old web-fetch README TBD. Downloadable 33.05MB checkpoint https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset has no license card/source split metadata. SHA256 41c250587af2e947dc3e265cd6e7aa7de230048d408e93c53d8348d63cf75896. Private evaluation 12 requests yielded +7.08dB mean full4s, +6.82dB 2s; use Resemblyzer 256D, not TitaNet. FFT/complex state export and global normalization mean not streaming as-is. Do not ship its weights without permission.

## Listening sequence
listening-clean.wav: 0-4s mix, 5-9s target A clean reference, 10-14s extracted A, 15-19s extracted B. 1s gaps.
listening-reverb.wav: 0-4s mix, 5-9s reverberant target reference, 10-14s extraction. Public synthetic fixtures, not live phone audio. Human listening verdict still required.

## Android feasibility wiring
Separate application ID com.akashrajeev.voicebeam.lab.tse, can coexist with stable
lab APK. Separation OFF by default; settings must explicitly enable it. Learn a
locked target first on the listening screen. Experimental mode uses raw 2-second
blocks with no GTCRN before separation. It plays/records buffered selected output
only on the existing confirmed media-earphone route. No target template or low
cosine/margin -> silence. Threshold controls are uncalibrated; defaults .35 cosine
and .10 margin are starting experiment policy, not validated acceptance rates.
TitaNet operations serialize to avoid native-model concurrent use. This can
increase delay while voice enrollment/scoring also runs. Memory telemetry is Java
heap only, not full native ONNX process memory. p50/p95 uses last 50 blocks.
Source changes during a computation invalidate its output block. Input/output
queues each hold one pending block and count drops. A compute error disables the
worker into silence and records error class. Peak clamp is .95. This is a PRIVATE
phone feasibility experiment, not cleared commercial weights or final live hearing.

## Updated worker validation
See WORKER_UPDATE.md, frozen-metrics.json, validate_frozen.py and safe export parity.
Host evaluation with Android-like frozen enrollment supports .35 cosine / .10
margin as experimental initial controls: clean 23/24 accepted, noisy 10/24,
absent/other-only/silence 0/24. Small synthetic evaluation, not phone proof or
calibrated field acceptance guarantee. The defaults already match #12 eb550fb.
