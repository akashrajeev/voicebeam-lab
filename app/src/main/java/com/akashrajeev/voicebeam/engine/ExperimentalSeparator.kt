package com.akashrajeev.voicebeam.engine

import android.content.Context
import android.os.SystemClock
import android.os.Debug
import android.os.BatteryManager
import android.os.PowerManager
import android.os.Build
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
import kotlin.math.sqrt

/** Private phone probe. Continuity is NOT evidence that the enrolled voice is present. */
class ExperimentalSeparator(
    app: Context, private val voice: VoicePrint, private val centroid: () -> FloatArray?,
    private val threshold: () -> Float, private val margin: () -> Float,
    private val rawSelection: () -> Boolean,
    contextMs: Int = 1000, hopMs: Int = 250,
    private val autoTarget:()->Int?={null}, private val autoEligible:()->Boolean={false},
) {
    private val battery = app.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val power = app.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val context = contextMs * 16
    private val hop = hopMs * 16
    private data class Packet(val audio: FloatArray, val template: FloatArray?, val endSample: Long, val capturedNs: Long,val reference:FloatArray?=null,val targetId:Int?=null,val isolated:Boolean=false)
    private val automatic=AutoVoiceReference()
    private fun targetTemplate()=centroid() ?: automatic.centroid
    @Volatile var selectedOutput=false;private set
    private var referencePending:FloatArray?=null
    private var referenceTarget:Int?=null
    private val input = DropOldestQueue<Packet>(1)
    private val output = DropOldestQueue<Packet>(1)
    private val running = AtomicBoolean(true)
    private val busy = AtomicBoolean(false)
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2); setInterOpNumThreads(1);setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT) }
    private val session: OrtSession
    private val ring = FloatArray(context)
    private var write = 0
    private var hopFill = 0
    private var captured = 0L
    private var playing: Packet? = null
    private var playOffset = 0
    private var previousSample = 0f
    private var previousTemplate: FloatArray? = null
    private var droppedBusy = 0L
    private var staleOutput = 0L
    private val times = ArrayList<Long>()
    private val startNs = SystemClock.elapsedRealtimeNanos()
    private val worker: Thread
    init {
        require(context == 16000 || context == 32000)
        require(hop in 4000..32000 && hop <= context)
        val name = if(context == 16000) "convtasnet-1s-int8.onnx" else "convtasnet-2s-int8.onnx"
        val path = File(app.cacheDir, name)
        if (!path.exists()) app.assets.open("models/$name").use { src -> path.outputStream().use { src.copyTo(it) } }
        session = env.createSession(path.absolutePath, options)
        Diagnostics.event("TSE enabled model=int8 identityHz=1 failOpenNoMute=true contextMs=$contextMs hopMs=$hopMs nativeORT="+env.version)
        worker = Thread({ loop() }, "vb-tse").also { it.start() }
    }
    /** Raw mic only. Zero-pad initial context. No pending compute when worker is behind. */
    fun process(raw: FloatArray, out: FloatArray): Boolean {
        if(referenceTarget!=autoTarget()){referencePending=null;referenceTarget=autoTarget()}
        automatic.setTarget(autoTarget())
        if(centroid()==null && automatic.centroid==null){automatic.collect(raw,autoEligible())?.let{referencePending=it}}
        for(v in raw) {
            ring[write] = v; write = (write+1)%context; captured++; hopFill++
            if(hopFill == hop) {
                hopFill=0
                if(running.get() && busy.compareAndSet(false,true)) {
                    val window=FloatArray(context) { k -> ring[(write+k)%context] }
                    input.offer(Packet(window,targetTemplate(),captured,SystemClock.elapsedRealtimeNanos(),referencePending,autoTarget()));referencePending=null
                } else droppedBusy++
            }
        }
        out.fill(0f);selectedOutput=false
        if(targetTemplate()==null) {
            for(k in out.indices) out[k]=raw.getOrElse(k){0f}.coerceIn(-.95f,.95f)
            Diagnostics.tseState="AUTO REFERENCE ${automatic.phrases}/3: wait for locked face solo speech; raw passthrough"
            return true
        }
        var written=0
        while(written < out.size) {
            if(playing == null || playOffset >= playing!!.audio.size) {
                playing=output.poll(0); playOffset=0
                val packet=playing
                if(packet != null) {
                    val age=SystemClock.elapsedRealtimeNanos()-packet.capturedNs
                    if(age > (context+hop)*1000000000L/16000) { playing=null; staleOutput++ }
                    else Diagnostics.event("TSE softwareCaptureToConsumeMs="+age/1000000+" excludesAudioTrackBluetooth=true")
                }
            }
            val packet=playing ?: break
            if(packet.template == null || packet.template !== targetTemplate()) { playing=null; previousSample=0f; continue }
            if(previousTemplate !== packet.template) { previousSample=0f; previousTemplate=packet.template }
            // A muted packet remains exact silence, including its seam.
            if(playOffset==0 && packet.audio.all { it == 0f }) previousSample=0f
            selectedOutput=packet.isolated
            val n=minOf(out.size-written,packet.audio.size-playOffset)
            for(k in 0 until n) {
                val pos=playOffset+k
                val v=packet.audio[pos]
                out[written+k]=if(pos<80) previousSample*(1f-pos/80f)+v*(pos/80f) else v
            }
            playOffset+=n; written+=n
            if(playOffset == packet.audio.size && n>0) previousSample=out[written-1]
        }
        return written>0
    }
    private fun correlation(a:FloatArray, aStart:Int, b:FloatArray, size:Int):Float {
        var dot=0.0; var aa=0.0; var bb=0.0
        for(i in 0 until size) { val x=a[aStart+i].toDouble(); val y=b[i].toDouble(); dot+=x*y; aa+=x*x; bb+=y*y }
        return (dot/(sqrt(aa*bb)+1e-8)).toFloat()
    }
    private fun loop() {
        var previous:Packet?=null
        var stickyEvidence=0f
        var identity=SlowIdentity(hop,context)
        var identityTemplate:FloatArray?=null
        val presence=SeparationPresence()
        try {
            while(running.get()) {
                val packet=input.poll(200) ?: continue
                try {
                    if(packet.template==null && packet.reference!=null && packet.targetId==autoTarget()){
                        val learned=automatic.accept(voice.embed(packet.reference))
                        Diagnostics.event("TSE AUTO_REFERENCE phrases=${automatic.phrases} ready=$learned provisional=true")
                    }
                    val template=packet.template
                    if(template == null || template !== targetTemplate()) { previous=null; stickyEvidence=0f; Diagnostics.tseState="UNCERTAIN: no frozen target"; continue }
                    val start=SystemClock.elapsedRealtimeNanos()
                    val sources:Array<FloatArray> = OnnxTensor.createTensor(env,FloatBuffer.wrap(packet.audio),longArrayOf(1,context.toLong())).use { tensor ->
                        session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            val value=result[0].value as Array<Array<FloatArray>>
                            value[0].map { it.copyOf() }.toTypedArray()
                        }
                    }
                    val sepEnd=SystemClock.elapsedRealtimeNanos()
                    for(source in sources) for(k in source.indices) source[k]=if(source[k].isFinite()) source[k].coerceIn(-1f,1f) else 0f
                    if(identityTemplate !== template){identity=SlowIdentity(hop,context);presence.reset();identityTemplate=template;previous=null}
                    val prevShift=previous?.let{(packet.endSample-it.endSample).toInt()} ?: hop
                    val update=identity.update(sources,prevShift){source->voice.embed(source)?.let{VoiceMatch.cosine(it,template)} ?: -1f}
                    val scores=update.scores
                    val speakerEnd=SystemClock.elapsedRealtimeNanos()
                    val freshBest=if(scores[0]>=scores[1]) 0 else 1
                    var selectedIndex=freshBest
                    val prev=previous
                    val shift=if(prev == null) context else (packet.endSample-prev.endSample).toInt()
                    if(prev != null && prev.template === template && shift in 1 until context) {
                        val continuity=sources.map { correlation(prev.audio,shift,it,context-shift) }
                        val linked=if(continuity[0]>=continuity[1]) 0 else 1
                        // Accumulate evidence for linked waveform, never use it to assert presence.
                        stickyEvidence=.75f*stickyEvidence+.25f*scores[linked]
                        if(abs(scores[0]-scores[1])<margin() || scores[freshBest]<stickyEvidence+.10f) selectedIndex=linked
                    } else stickyEvidence=scores[freshBest]
                    val selected=sources[selectedIndex]
                    previous=Packet(selected.copyOf(),template,packet.endSample,packet.capturedNs)
                    val accepted=template === targetTemplate() && scores.all { it.isFinite() } && scores[selectedIndex]>threshold() &&
                        scores[selectedIndex]-scores[1-selectedIndex]>margin()
                    val raw=rawSelection()
                    val decision=presence.decide(scores,update.fresh,SystemClock.elapsedRealtime(),template === targetTemplate(),threshold(),margin())
                    // At1Hz a negative can outlive target entry. Never mute in this candidate.
                    val strongOther = false
                    val emitted=if((raw || (accepted && decision==SeparationPresence.Decision.SELECT)) && template === targetTemplate()) selected.copyOfRange(context-hop,context)
                        else if(!strongOther) packet.audio.copyOfRange(context-hop,context) else FloatArray(hop)
                    var peak=0f; for(v in emitted) peak=maxOf(peak,abs(v))
                    if(peak>.95f) for(k in emitted.indices) emitted[k]*=.95f/peak
                    if(running.get()) output.offer(Packet(emitted,template,packet.endSample,packet.capturedNs,isolated=accepted && decision==SeparationPresence.Decision.SELECT && !raw))
                    val ms=(speakerEnd-start)/1000000
                    times.add(ms); if(times.size>1200) times.removeAt(0)
                    val sorted=times.sorted(); val p50=sorted[(sorted.size-1)/2]; val p95=sorted[(sorted.size-1)*95/100]
                    val runtime=Runtime.getRuntime()
                    Diagnostics.tseState=if(raw) "RAW PROBE: target presence unverified" else if(accepted) "Provisional match" else if(strongOther) "Provisional non-target: muted" else "UNCERTAIN: raw mix passthrough"
                    Diagnostics.event("TSE sepMs="+(sepEnd-start)/1000000+" speakerMs="+(speakerEnd-sepEnd)/1000000+
                        " p50Ms=$p50 p95Ms=$p95 sustainSec="+(speakerEnd-startNs)/1000000000+" computeWithin200ms="+(p95<=200)+
                        " nativeHeapMB="+Debug.getNativeHeapAllocatedSize()/1048576+" usedHeapMB="+(runtime.totalMemory()-runtime.freeMemory())/1048576+
                        " batteryPct="+battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)+
                        " thermal="+(if(Build.VERSION.SDK_INT>=29) power.currentThermalStatus else -1)+
                        " backlog="+input.size+" dropsBusy=$droppedBusy dropsOutput="+output.dropped+" staleOutput=$staleOutput"+
                        " model=int8 threads=2 identityHz=1 identityFresh="+update.fresh+" cosine0="+scores[0]+" cosine1="+scores[1]+" stickyEvidence=$stickyEvidence raw=$raw accepted=$accepted peak=$peak")
                } finally { busy.set(false) }
            }
        } catch(t:Throwable) { Diagnostics.tseState="ERROR: "+t.javaClass.simpleName; Diagnostics.event("TSE error="+t.javaClass.simpleName); running.set(false) }
        finally { session.close(); options.close() }
    }
    fun stop() { running.set(false); worker.join(5000); playing=null; input.clear(); output.clear(); Diagnostics.tseState="OFF" }
}
