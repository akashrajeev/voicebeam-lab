# Private phone feasibility probe

Install separately from stable. Start with a low earphone volume. No speaker route
changed. Enable experiment in Settings only after clean target enrollment.
Mode 1s/250ms uses sliding zero-padded context and newest 250ms output. Mode
2s/500ms is quality fallback; 2s blocks preserves first feasibility behavior.
Do not test near traffic or rely on this for situational hearing.

Gated mode is default: .35 cosine/.10 margin per window, deliberately uncertain
mute. On host it rejects most real target-present 1s windows. RAW probe keeps
speaker assignment through weak evidence using waveform overlap and accumulated
identity scores, but DOES NOT prove target is present. It can play an impostor
when target is absent. No reservoir-based presence claim. UI labels both modes.

Log export: Settings -> Share log. Contains measurements only, no audio/captions.
Record clean/raw audio explicitly if desired; saves locally, never auto-uploads.
Test two voices together, target absent, target alone, other alone, target returns,
same-sex pair, silence and five-minute sustain. Listen for wrong-speaker switches,
seams, missing syllables, delay, feedback and heat. Log compute p50/p95 (1200 recent
blocks), drops, Java/native heap, battery and Android thermal status. Target p95
<=200ms for 250ms hop is proposed, not achieved yet. Busy hops skipped entirely;
zero pending compute backlog, no silently growing latency. Playback queue one hop.
Software capture-to-consume latency excludes mic hardware, AudioTrack, Bluetooth
buffers and acoustic delay; no end-to-end latency claim. If slow, choose fallback.

Native runtime single-provider packaging must still be verified by phone: start
models, enroll, enable experiment, get successful TSE compute rows. Build alone
doesn't prove sherpa or ORT JNI can load together. Errors display/log class.

No UI screenshot/device verification yet. Both mode and native core are private
experiments. Source/model notices and CC-BY-SA 3/4 conflict in LICENSE_NOTES.md.

#14 actual phone log (Oct8 02:24): 300 retained events, 135 audio rows,
voiceMatch null135/135, VAD false115/135. Target enrollment began but no phrase
completed, progress max sampled0.00533 then resets. TSE rows absent so no TSE
execution established. Enrollment repair: deliberately capture raw mic for three
3-second buffers, minimum phrase RMS .005, no lip/VAD reset. Other speaker must
stay silent; raw capture can enroll the wrong voice if protocol violated.
Experimental incomplete/uncertain states pass raw mic/mix WITHOUT boost, visibly
no isolation. Only complete-template low-score two-source negative can mute.
This is a usability policy, not validated absence detector; false accepts remain.
