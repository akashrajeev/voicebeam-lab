#!/usr/bin/env bash
set -euo pipefail
mkdir -p smoke-result
adb emu virtualscene-image wall "$PWD/scripts/emulator-smoke/face-poster.png" | tee smoke-result/camera-console.txt
python scripts/emulator-smoke/inject.py scripts/emulator-smoke/input.wav > smoke-result/injection.txt 2>&1 &
inject_pid=$!
set +e
adb shell am instrument -w -e class com.akashrajeev.voicebeam.EmulatorSmokeTest com.akashrajeev.voicebeam.lab.test/androidx.test.runner.AndroidJUnitRunner | tee smoke-result/instrumentation.txt
rc=${PIPESTATUS[0]}
set -e
adb logcat -d > smoke-result/logcat.txt
adb pull /sdcard/Android/data/com.akashrajeev.voicebeam.lab/files/emulator-smoke.png smoke-result/ || true
wait "$inject_pid" || true
[ "$rc" = 0 ] && grep -q 'OK (1 test)' smoke-result/instrumentation.txt
