# SEP-6 Pixel8, API34, October8
Commit7c852c5. Passed1 test,202sec is execution success only.
Device logs: https://console.firebase.google.com/project/voicebeam-lab-tests/testlab/histories/bh.61d189f6574727b8/matrices/7704368116440334746/executions/bs.7c10500b10f543b0/logs
Log file: https://storage.cloud.google.com/test-lab-1fbq0wujb51ku-m7it5v5bfmccx/2026-10-08_09:28:45.886707_lNkg/shiba-34-en-portrait/logcat?authuser=0

Int8 CPU2threads identity1Hz:
context/hop | overlap p50/p95ms | selected-separator SI-SDRi | projected misses
1000/250 |149/271|12.7155dB|10/48
2000/500 |329/436|13.1223dB|0/24
These SI-SDRi rows are NOT fail-open production output quality; all source-selected hops are scored regardless of RAW/MUTE decision.
Paced separator6sec:134/500 postwarm10ms frames missing(1s);148/400(2s). Test caller had no microphone/playback/camera. Production AudioPipeline ignored process=false, so missing frames became silence. SEP-7 fixes this with raw fallback but cannot inherit phone validation.
Auto-reference3x2sec scripted solo speech converged; real camera association unverified.
Other-only selectedHops0 at both contexts; proposed negative-mute counts are NOT actual production mute.1s target-entry policy had2 stale mute decisions after entry; production negative mute disabled. First selected4sec into fixture,2sec after target entered.
2s other-only p95597ms,misses5/24; silence565ms,9/24; target-entry636ms,12/24. Short tests/order effects do not establish heat endurance.
In-memory Diagnostics ring was NOT exported in this run, so drops/heap/thermal unavailable. SEP-7 instrumentation exports it. Bluetooth/acoustic latency unmeasured.
Target-only SI-SDRi subtracts near-perfect clean input and is not a useful clean-quality report. SEP-7 also logs absolute selected-output SI-SDR.
Verdict: no default/stable ENH merge; retain private defaultOFF experiment.
