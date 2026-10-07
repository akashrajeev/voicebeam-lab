# Silent live monitor investigation

Base APK: 4c9811f on lab/hardening-2026-10-07.

The output wrote nothing until AudioTrack.routedDevice was a headphone route.
A stream may not establish its route until it consumes buffers. This can leave
startup stuck: no route means no write, and no write means no active route.
The same guard exists in public main, so this is a latent defect, not a proven
new regression. No phone logs were available to identify the user's exact cause.

Fix: prime the track with silence and keep silence flowing on unknown/speaker
routes. Only an actual headphone route gets live microphone samples. USB audio
adapters are recognized consistently. Negative read/write results and audio
thread exceptions are reported; the UI no longer stays silently listening after
an audio failure. The pipeline's Silero instance is released at thread exit.

Gate arithmetic and defaults do not fully mute speech: quietOthers defaults to
0.8 and boost to +12 dB. Moonshine and TitaNet do not directly control playback.
Silero errors can stop the audio thread, which previously only logged them.
The DSP hardening math matches the earlier inline implementation.
Microphone permission remains declared. AudioTrack does not require an added
Bluetooth scan/connect permission just to play via an existing media route.

Four policy tests cover unknown-route priming, headphone speech, mute, and route
changes. They do not simulate Android AudioTrack or prove behavior on iQOO.

Device checks still needed: earphone route (wired/USB/Bluetooth vs speaker),
media volume, model-ready state, moving input meter and captions, and logcat
VoiceBeamAudio/VoiceBeamModels/VoiceBeamEngine output. Try connecting headphones
before start, then after start, then unplugging (speech must not reach speaker).
This patch is not an overlap separator and makes no isolation-quality claim.

Android routing contract:
https://developer.android.com/reference/android/media/AudioRouting#getRoutedDevice()

## Local diagnostics
Settings > Live diagnostics updates once a second. The latest 300 technical
events are held in RAM, with route type, audio RMS/gain/VAD/gate, write results,
denoiser/VAD/gate/playback/ASR/speaker timings and caption backlog. Model and
audio failures log only exception class, not arbitrary exception content.
No audio samples, captions, face data, device names or identifiers are included.
There is no remote feed or upload endpoint. Settings > Share log exports a
text snapshot through Android's chooser; select WhatsApp and the Instinct chat.
Clear log clears RAM. Restarting the process also clears it. A previous exported
snapshot can remain in app cache until replaced or the cache is cleared.
