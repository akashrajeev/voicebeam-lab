package com.akashrajeev.voicebeam.engine

import android.content.Context
import android.os.SystemClock
import android.util.Log
import ai.onnxruntime.*
import com.akashrajeev.voicebeam.core.VoiceMatch
import com.akashrajeev.voicebeam.core.WavWriter
import com.akashrajeev.voicebeam.ml.VoicePrint
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.*

/** Public fixture only. No microphone, camera, private data or playback. */
object SeparationSelfTest {
    private data class Variant(val threads:Int,val slow:Boolean,val kind:String="fp32",val provider:String="CPU")
    private val variants=listOf(Variant(2,false),Variant(2,true),Variant(3,true),Variant(4,true),
        Variant(2,true,"int8"),Variant(4,true,"int8"),Variant(2,true,"fp16"),
        Variant(2,true,"fp32","NNAPI"),Variant(2,true,"int8","NNAPI"),Variant(2,true,"fp32","XNNPACK"),Variant(2,true,"int8","XNNPACK"))
    @Volatile var status = "Not run"
    @Volatile private var busy = false
    @Synchronized fun start(ctx:Context) {
        if(busy) return
        busy=true; status="Running public fixture self-test"
        Thread({ try { run(ctx) } catch(t:Throwable) { status="FAILED: "+t.javaClass.simpleName+": "+t.message; log(status) } finally {busy=false} },"vb-selftest").start()
    }
    private fun log(s:String) { Diagnostics.event("SELFTEST $s"); Log.i("VoiceBeamSelfTest",s) }
    private fun load(ctx:Context,name:String):FloatArray {
        val file=File(ctx.cacheDir,name)
        ctx.assets.open("selftest/$name").use { src->file.outputStream().use {src.copyTo(it)} }
        val (samples,sr)=WavWriter.read(file);check(sr==16000);return samples
    }
    private fun repeatNormalize(a:FloatArray,size:Int):FloatArray {
        val rms=sqrt(a.fold(0.0){e,v->e+v*v}/a.size).toFloat()
        return FloatArray(size){a[it%a.size]*(.06f/max(rms,1e-6f))}
    }
    private fun sdr(y:FloatArray,x:FloatArray):Double {
        val ym=y.average();val xm=x.average();var dot=0.0;var xx=0.0
        for(i in y.indices){dot+=(y[i]-ym)*(x[i]-xm);xx+=(x[i]-xm).pow(2)}
        val gain=dot/(xx+1e-12);var proj=0.0;var err=0.0
        for(i in y.indices){val p=gain*(x[i]-xm);proj+=p*p;err+=(y[i]-ym-p).pow(2)}
        return 10*log10((proj+1e-12)/(err+1e-12))
    }
    /** Runnable headlessly. Exact results saved as text + logcat and explicit user Share log. */
    fun run(ctx:Context):String {
        val report=StringBuilder("Public synthetic fixture, NOT room/Bluetooth test\n")
        val voice=VoicePrint(ctx.assets)
        try {
            val enrollment=load(ctx,"1089-134686-0013.wav")
            val template=voice.embed(enrollment.copyOfRange(0,minOf(enrollment.size,48000))) ?: error("TitaNet enrollment failed")
            log("TitaNet modelLoad=PASS embeddingDims="+template.size)
            val a=repeatNormalize(load(ctx,"1089-134686-0002.wav"),96000)
            val b=repeatNormalize(load(ctx,"1221-135767-0005.wav"),96000)
            val mix=FloatArray(a.size){a[it]+b[it]}
            val env=OrtEnvironment.getEnvironment()
            for(context in listOf(16000,32000)) for(variant in variants) {
                val threads=variant.threads;val slow=variant.slow
                try {
                val hop=if(context==16000) 4000 else 8000
                val name="convtasnet-${context/16000}s-${if(variant.kind=="fp32") "safe" else variant.kind}.onnx"
                val path=File(ctx.cacheDir,name)
                if(!path.exists())ctx.assets.open("models/$name").use {src->path.outputStream().use {src.copyTo(it)}}
                val opts=OrtSession.SessionOptions().apply{setIntraOpNumThreads(threads);setInterOpNumThreads(1);setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)}
                if(variant.provider=="NNAPI")opts.addNnapi()
                if(variant.provider=="XNNPACK")opts.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                val loadStart=SystemClock.elapsedRealtimeNanos()
                val session=env.createSession(path.absolutePath,opts)
                val loadMs=(SystemClock.elapsedRealtimeNanos()-loadStart)/1000000
                log("context=$context modelLoad=PASS loadMs=$loadMs nativeORT="+env.version)
                try {
                    val output=FloatArray(mix.size);val compute=ArrayList<Long>();val sepTimes=ArrayList<Long>();val idTimes=ArrayList<Long>();val refreshTimes=ArrayList<Long>();var missed=0;var uncertain=0
                    val identity=SlowIdentity(hop,context)
                    var previous:FloatArray?=null;var last=0f
                    for(end in hop..mix.size step hop) {
                        val window=FloatArray(context){k->val pos=end-context+k;if(pos>=0)mix[pos] else 0f}
                        val start=SystemClock.elapsedRealtimeNanos()
                        val sources:Array<FloatArray> = OnnxTensor.createTensor(env,FloatBuffer.wrap(window),longArrayOf(1,context.toLong())).use{tensor->
                            session.run(mapOf(session.inputNames.first() to tensor)).use{result->
                                @Suppress("UNCHECKED_CAST") val value=result[0].value as Array<Array<FloatArray>>
                                value[0].map{it.copyOf()}.toTypedArray()
                            }
                        }
                        check(sources.all{src->src.all{it.isFinite()}}){"Nonfinite separator output"}
                        val sepEnd=SystemClock.elapsedRealtimeNanos()
                        val update=if(slow)identity.update(sources){src->voice.embed(src)?.let{VoiceMatch.cosine(it,template)} ?: -1f} else null
                        val scores=update?.scores ?: sources.map{src->voice.embed(src)?.let{VoiceMatch.cosine(it,template)} ?: -1f}
                        val idEnd=SystemClock.elapsedRealtimeNanos()
                        sepTimes.add((sepEnd-start)/1000000);idTimes.add((idEnd-sepEnd)/1000000)
                        var best=if(scores[0]>=scores[1])0 else 1
                        val prev=previous
                        if(prev!=null && abs(scores[0]-scores[1])<.10f) {
                            val corr=sources.map{src->var d=0.0;var aa=0.0;var bb=0.0
                                for(k in 0 until context-hop){val x=prev[k+hop].toDouble();val y=src[k].toDouble();d+=x*y;aa+=x*x;bb+=y*y};d/(sqrt(aa*bb)+1e-8)}
                            best=if(corr[0]>=corr[1])0 else 1
                        }
                        previous=sources[best].copyOf()
                        if(scores[best]<=.35f || scores[best]-scores[1-best]<=.10f)uncertain++
                        val elapsed=(SystemClock.elapsedRealtimeNanos()-start)/1000000;compute.add(elapsed)
                        if(elapsed>hop/16)missed++
                        if(update?.fresh==true)refreshTimes.add(elapsed)
                        for(k in 0 until hop){val v=sources[best][context-hop+k].coerceIn(-.95f,.95f);output[end-hop+k]=if(k<80)last*(1-k/80f)+v*(k/80f) else v}
                        last=output[end-1]
                    }
                    val times=compute.sorted();val p50=times[(times.size-1)/2];val p95=times[(times.size-1)*95/100]
                    val improvement=sdr(output.copyOfRange(context,output.size),a.copyOfRange(context,a.size))-sdr(mix.copyOfRange(context,mix.size),a.copyOfRange(context,a.size))
                    fun percentile(x:List<Long>,pct:Int):Long {val a=x.sorted();return if(a.isEmpty())0 else a[(a.size-1)*pct/100]}
                    val row="kind=${variant.kind} provider=${variant.provider} threads=$threads identityHz=${if(slow)1 else 1000/(hop/16)} sepP95Ms=${percentile(sepTimes,95)} idP95Ms=${percentile(idTimes,95)} refreshP95Ms=${percentile(refreshTimes,95)} contextMs=${context/16} hopMs=${hop/16} p50Ms=$p50 p95Ms=$p95 SI_SDRi=$improvement projectedDeadlineMisses=$missed/${compute.size} uncertainHops=$uncertain actualQueueDrops=NA offlineSequential=true"
                    log(row);report.append(row).append('\n')
                }finally{session.close();opts.close()}
                } catch(t:Throwable) {val failure="VARIANT_FAILED contextMs=${context/16} kind=${variant.kind} provider=${variant.provider} threads=$threads error=${t.javaClass.simpleName}: ${t.message}";log(failure);report.append(failure).append('\n')}
            }
            status="PASS native models ran. Quality/speed: see log. Not room validated."
            log(status);report.append(status)
            File(ctx.getExternalFilesDir(null),"voicebeam-selftest.txt").writeText(report.toString())
            return report.toString()
        }finally{voice.release()}
    }
}
