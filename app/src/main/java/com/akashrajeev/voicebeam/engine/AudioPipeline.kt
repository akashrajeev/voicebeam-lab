package com.akashrajeev.voicebeam.engine

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioDeviceInfo
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.MicrophoneDirection
import android.os.Build
import android.os.Process
import android.os.SystemClock
import android.util.Log
import com.akashrajeev.voicebeam.core.EnergyVad
import com.akashrajeev.voicebeam.core.GateInputs
import com.akashrajeev.voicebeam.core.TargetGate
import com.akashrajeev.voicebeam.core.WavWriter
import com.akashrajeev.voicebeam.ml.AudioModels
import com.akashrajeev.voicebeam.ml.SAMPLE_RATE
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * Mic -> noise removal -> target gate -> earphones, with side outputs for
 * captions, the voice fingerprint and recording. Runs on its own thread.
 * When [debugFeed] is set (debug builds only), bundled samples replace the mic.
 */
class AudioPipeline(
    private val models: AudioModels,
    private val app: android.content.Context,
    private val signals: () -> GateInputs,   // latest vision + voice info (voiceActive is filled here)
    private val onFrame: (FrameInfo) -> Unit,
) {
    data class FrameInfo(val level: Float, val gain: Float, val probability: Float, val voiceActive: Boolean)
    data class CaptionBlock(val samples: FloatArray, val captureMs: Long, val probability: Float, val voiceActive: Boolean)
    private var capturedSamples = 0L
    var droppedCaptionBlocks = 0L
        private set

    @Volatile var quietOthers = 0.8f
    @Volatile var boostDb = 12f
    @Volatile var denoiseMix = 1f          // 0 = raw, 1 = fully denoised
    @Volatile var monitorEnabled = true    // hard-limited to an actual headphone route
    @Volatile var rawWriter: WavWriter? = null
    @Volatile var cleanWriter: WavWriter? = null
    /** Debug builds only: when set, called with the frame size to produce mic input. */
    @Volatile var debugFeed: ((Int) -> FloatArray)? = null

    /** Gated clean audio for captions (consumer: caption thread). */
    val asrQueue = ArrayBlockingQueue<CaptionBlock>(400)
    /** Denoised audio + whether the locked face was clearly talking (consumer: voice-print thread). */
    val voiceQueue = ArrayBlockingQueue<Pair<FloatArray, Float>>(400)

    private val gate = TargetGate(frameMs = (models.denoiser.frameShift.takeIf { it > 0 } ?: 256) * 1000f / SAMPLE_RATE)
    private val vad = com.akashrajeev.voicebeam.ml.NeuralVad(app.assets)
    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private var dbgFrames = 0L

    @SuppressLint("MissingPermission")
    fun start(useSceneMic: Boolean) {
        if (running.getAndSet(true)) return
        capturedSamples = 0L; droppedCaptionBlocks = 0L; vad.reset()
        thread = Thread({ loop(useSceneMic) }, "vb-audio").also { it.start() }
    }

    fun stop() {
        running.set(false)
        thread?.join(1500)
        thread = null
    }

    @SuppressLint("MissingPermission")
    private fun loop(useSceneMic: Boolean) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val frameShift = models.denoiser.frameShift.takeIf { it > 0 } ?: 256
        val dbg = debugFeed
        val rec = if (dbg != null) null else {
            val minRec = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
            val source = if (useSceneMic) MediaRecorder.AudioSource.CAMCORDER else MediaRecorder.AudioSource.VOICE_RECOGNITION
            try {
                AudioRecord(source, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT, maxOf(minRec, frameShift * 8))
            } catch (t: Throwable) {
                AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT, maxOf(minRec, frameShift * 8))
            }
        }
        if (rec != null && Build.VERSION.SDK_INT >= 29) {
            try {
                rec.setPreferredMicrophoneDirection(
                    if (useSceneMic) MicrophoneDirection.MIC_DIRECTION_AWAY_FROM_USER else MicrophoneDirection.MIC_DIRECTION_UNSPECIFIED
                )
            } catch (_: Throwable) {}
        }
        val track = if (dbg != null) null else {
            val minPlay = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
            AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setSampleRate(SAMPLE_RATE).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setBufferSizeInBytes(maxOf(minPlay, frameShift * 4 * 4))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
                .build()
        }
        // Prefer a physical earphone route; never open monitor audio to the speaker.
        val headset = (app.getSystemService(android.content.Context.AUDIO_SERVICE) as android.media.AudioManager)
            .getDevices(android.media.AudioManager.GET_DEVICES_OUTPUTS).firstOrNull {
                it.type in setOf(AudioDeviceInfo.TYPE_WIRED_HEADPHONES, AudioDeviceInfo.TYPE_WIRED_HEADSET,
                    AudioDeviceInfo.TYPE_USB_HEADSET, AudioDeviceInfo.TYPE_BLUETOOTH_A2DP,
                    AudioDeviceInfo.TYPE_BLUETOOTH_SCO, AudioDeviceInfo.TYPE_BLE_HEADSET)
            }
        if (headset != null) track?.preferredDevice = headset
        val input = FloatArray(frameShift)
        try {
            rec?.startRecording()
            track?.play()
            models.denoiser.reset()
            gate.reset()
            var fedFrames = 0L
            val feedStart = SystemClock.uptimeMillis()
            while (running.get()) {
                if (dbg != null) {
                    val f = dbg(frameShift)
                    System.arraycopy(f, 0, input, 0, minOf(f.size, frameShift))
                    fedFrames++
                    // Pace to real time so captions, lips and the gate line up.
                    val due = feedStart + fedFrames * frameShift * 1000L / SAMPLE_RATE
                    val wait = due - SystemClock.uptimeMillis()
                    if (wait > 0) Thread.sleep(wait)
                } else {
                    var read = 0
                    while (read < frameShift && running.get()) {
                        val n = rec!!.read(input, read, frameShift - read, AudioRecord.READ_BLOCKING)
                        if (n <= 0) break
                        read += n
                    }
                    if (read < frameShift) continue
                }
                capturedSamples += input.size
                rawWriter?.write(input)

                val denoised = try { models.denoiser.process(input.copyOf()) } catch (t: Throwable) { input.copyOf() }
                val n = minOf(denoised.size, input.size)
                val clean = FloatArray(n)
                val mix = denoiseMix
                if (n == 0) continue
                // denoiser output may lag input by a frame; mixing uses aligned sizes
                for (k in 0 until n) clean[k] = mix * denoised[k] + (1f - mix) * input[input.size - n + k]

                val voice = vad.isVoice(clean)
                val s = signals()
                gate.quietOthers = quietOthers
                val g = gate.process(s.copy(voiceActive = voice))
                val boost = TargetGate.dbToLinear(boostDb)
                val gated = FloatArray(n)
                val out = FloatArray(n)
                var e = 0f
                for (k in 0 until n) {
                    gated[k] = clean[k] * g
                    out[k] = (gated[k] * boost).coerceIn(-1f, 1f)
                    e += clean[k] * clean[k]
                }
                cleanWriter?.write(gated)
                // Speaker playback of a boosted live microphone causes a runaway feedback loop.
                // The actual routed output, not merely a paired headset, must be safe.
                val routed = track?.routedDevice
                val headphoneRoute = routed != null && routed.type in setOf(
                    AudioDeviceInfo.TYPE_WIRED_HEADPHONES, AudioDeviceInfo.TYPE_WIRED_HEADSET,
                    AudioDeviceInfo.TYPE_USB_HEADSET, AudioDeviceInfo.TYPE_USB_DEVICE,
                    AudioDeviceInfo.TYPE_BLUETOOTH_A2DP, AudioDeviceInfo.TYPE_BLUETOOTH_SCO,
                    AudioDeviceInfo.TYPE_BLE_HEADSET,
                )
                if (monitorEnabled && headphoneRoute) track?.write(out, 0, n, AudioTrack.WRITE_NON_BLOCKING)
                // Recognition needs speech, not the gate's sometimes 80%-attenuated output.
                // Assign a caption to the target separately using the gate probability.
                val captionBlock = CaptionBlock(input.copyOf(), capturedSamples * 1000 / SAMPLE_RATE, gate.probability, voice)
                if (!asrQueue.offer(captionBlock)) { asrQueue.poll(); droppedCaptionBlocks++; asrQueue.offer(captionBlock) }
                if (voice) {
                    val v = Pair(clean, if (s.hasLock && s.othersSpeaking < 0.3f) s.lockedSpeaking else 0f)
                    if (!voiceQueue.offer(v)) { voiceQueue.poll(); voiceQueue.offer(v) }
                }
                onFrame(FrameInfo(sqrt(e / n), g, gate.probability, voice))
                if (com.akashrajeev.voicebeam.BuildConfig.DEBUG && ++dbgFrames % 400 == 0L) {
                    var er = 0f; for (k in 0 until input.size) er += input[k] * input[k]
                    Log.i("VoiceBeamAudio", "audiodbg dbg=" + (dbg != null) + " in=" + kotlin.math.sqrt(er / input.size) + " clean=" + kotlin.math.sqrt(e / n) + " g=" + g)
                }
            }
        } catch (t: Throwable) {
            Log.e("VoiceBeamAudio", "audio loop failed", t)
        } finally {
            try { rec?.stop() } catch (_: Throwable) {}
            rec?.release()
            try { track?.stop() } catch (_: Throwable) {}
            track?.release()
        }
    }
}
