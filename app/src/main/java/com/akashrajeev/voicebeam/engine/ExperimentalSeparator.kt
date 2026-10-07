package com.akashrajeev.voicebeam.engine

import android.content.Context
import android.os.SystemClock
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.akashrajeev.voicebeam.core.DropOldestQueue
import com.akashrajeev.voicebeam.core.VoiceMatch
import com.akashrajeev.voicebeam.ml.VoicePrint
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/** PRIVATE feasibility mode: blind two-source separation + enrolled target pick, not streaming. */
class ExperimentalSeparator(
    app: Context, private val voice: VoicePrint,
    private val centroid: () -> FloatArray?,
    private val threshold: () -> Float,
    private val margin: () -> Float,
) {
    private data class Packet(val audio: FloatArray, val template: FloatArray?)
    private val input = DropOldestQueue<Packet>(1) // one active worker + one pending block
    private val output = DropOldestQueue<Packet>(1)
    private val running = AtomicBoolean(true)
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2); setInterOpNumThreads(1) }
    private val session: OrtSession
    private val staging = FloatArray(32000)
    private var fill = 0
    private var playing: Packet? = null
    private var playOffset = 0
    private val times = ArrayList<Long>()
    private val worker: Thread
    init {
        val path = File(app.cacheDir, "convtasnet-2s-safe.onnx")
        if (!path.exists()) app.assets.open("models/convtasnet-2s-safe.onnx").use { src -> path.outputStream().use { src.copyTo(it) } }
        session = env.createSession(path.absolutePath, options)
        Diagnostics.event("TSE experimental_enabled bufferMs=2000 input=" + session.inputNames + " output=" + session.outputNames)
        worker = Thread({ loop() }, "vb-tse").also { it.start() }
    }
    /** Submit raw mic samples, consume buffered extraction; no identity -> silence. */
    fun process(raw: FloatArray, out: FloatArray): Boolean {
        var offset = 0
        while (offset < raw.size) {
            val n = minOf(raw.size-offset, staging.size-fill)
            System.arraycopy(raw, offset, staging, fill, n); offset += n; fill += n
            if (fill == staging.size) { input.offer(Packet(staging.copyOf(), centroid())); fill = 0 }
        }
        if (playing == null || playOffset >= playing!!.audio.size) { playing = output.poll(0); playOffset = 0 }
        val packet = playing
        out.fill(0f)
        if (packet == null || packet.template == null || packet.template !== centroid()) { playing=null; return false }
        val block = packet.audio
        val n = minOf(out.size, block.size-playOffset)
        System.arraycopy(block, playOffset, out, 0, n); playOffset += n
        return true
    }
    private fun loop() {
        try {
            while (running.get()) {
                val packet = input.poll(200) ?: continue
                val raw = packet.audio
                val currentTemplate = centroid()
                val target = currentTemplate?.copyOf()
                if (target == null || packet.template !== currentTemplate) { output.offer(Packet(FloatArray(32000), currentTemplate)); Diagnostics.event("TSE muted=no_target_template"); continue }
                val start = SystemClock.elapsedRealtimeNanos()
                val separated: Array<FloatArray> = OnnxTensor.createTensor(env, FloatBuffer.wrap(raw), longArrayOf(1,32000)).use { tensor ->
                    session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val value = result[0].value as Array<Array<FloatArray>>
                        value[0].map { it.copyOf() }.toTypedArray()
                    }
                }
                // A target change while processing must not play the previous person's block.
                if (centroid() !== currentTemplate) { output.offer(Packet(FloatArray(32000), currentTemplate)); continue }
                val sepEnd = SystemClock.elapsedRealtimeNanos()
                val scores = separated.map { source ->
                    // Protect extractor and playback from unexpected model scale/nonfinite values.
                    for (k in source.indices) source[k] = if (source[k].isFinite()) source[k].coerceIn(-1f,1f) else 0f
                    val embedding = voice.embed(source)
                    if (embedding == null) -1f else VoiceMatch.cosine(embedding,target)
                }
                val speakerEnd = SystemClock.elapsedRealtimeNanos()
                val best = if (scores[0] >= scores[1]) 0 else 1
                val accepted = currentTemplate === centroid() && running.get() &&
                    scores.all { it.isFinite() } && scores[best] >= threshold() && scores[best] - scores[1-best] >= margin()
                val selected = if (accepted) separated[best] else FloatArray(32000)
                var peak = 0f; for (v in selected) peak = maxOf(peak,abs(v))
                if (peak > 0.95f) for (k in selected.indices) selected[k] *= 0.95f/peak
                output.offer(Packet(selected,currentTemplate))
                val ms = (speakerEnd-start)/1000000
                times.add(ms); if (times.size>50) times.removeAt(0)
                val sorted=times.sorted()
                val p50=sorted[(sorted.size-1)/2]; val p95=sorted[((sorted.size-1)*95)/100]
                val rt=Runtime.getRuntime()
                Diagnostics.event("TSE sepMs="+(sepEnd-start)/1000000+" speakerMs="+(speakerEnd-sepEnd)/1000000+
                    " p50Ms="+p50+" p95Ms="+p95+" nativeHeapMB="+android.os.Debug.getNativeHeapAllocatedSize()/1048576+" usedHeapMB="+(rt.totalMemory()-rt.freeMemory())/1048576+
                    " droppedInput="+input.dropped+" droppedOutput="+output.dropped+" cosine0="+scores[0]+" cosine1="+scores[1]+
                    " threshold="+threshold()+" margin="+margin()+" accepted="+accepted+" peak="+peak)
            }
        } catch(t: Throwable) { Diagnostics.event("TSE error="+t.javaClass.simpleName); running.set(false) }
        finally { session.close(); options.close() }
    }
    fun stop() { running.set(false); worker.join(5000); playing=null; input.clear(); output.clear() }
}
