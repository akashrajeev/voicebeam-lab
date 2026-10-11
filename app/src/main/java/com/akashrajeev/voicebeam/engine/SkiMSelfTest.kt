package com.akashrajeev.voicebeam.engine
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.akashrajeev.voicebeam.core.*
import com.akashrajeev.voicebeam.ml.VoicePrint
import java.io.File
import kotlin.math.*

/** Public-fixture model quality plus paced exact production path, no actual mic/camera/playback. */
object SkiMSelfTest {
 @Volatile var status="Not run"
 @Volatile private var busy=false
 @Synchronized fun start(ctx:Context){if(busy)return;busy=true;status="Running causal SkiM fixtures";Thread({try{status=run(ctx)}catch(t:Throwable){status="FAILED "+t.javaClass.simpleName+": "+t.message;log(status)}finally{busy=false}},"vb-skim-test").start()}

 private fun log(s:String){Log.i("VoiceBeamSelfTest",s)}
 private fun sdr(y:FloatArray,x:FloatArray):Double {val ym=y.average();val xm=x.average();var dot=0.0;var xx=0.0;for(i in y.indices){dot+=(y[i]-ym)*(x[i]-xm);xx+=(x[i]-xm).pow(2)};val g=dot/(xx+1e-12);var p=0.0;var e=0.0;for(i in y.indices){val z=g*(x[i]-xm);p+=z*z;e+=(y[i]-ym-z).pow(2)};return 10*log10((p+1e-12)/(e+1e-12))}
 fun run(ctx:Context):String {
  fun load(name:String):FloatArray {val f=File(ctx.cacheDir,name);ctx.assets.open("selftest/$name").use{src->f.outputStream().use{src.copyTo(it)}};return WavWriter.read(f).first}
  fun norm(a:FloatArray):FloatArray {val rms=sqrt(a.fold(0.0){e,v->e+v*v}/a.size);return FloatArray(96000){(a[it%a.size]*.06/rms).toFloat()}}
  val a=norm(load("1089-134686-0002.wav"));val b=norm(load("1221-135767-0005.wav"));val rng=java.util.Random(31)
  val cases=linkedMapOf("overlap" to FloatArray(a.size){a[it]+b[it]},"target_only" to a,"other_only" to b,"silence" to FloatArray(a.size),"target_enters" to FloatArray(a.size){b[it]+if(it>=32000)a[it] else 0f},"noisy_overlap" to FloatArray(a.size){a[it]+b[it]+(rng.nextGaussian()*.02).toFloat()})
  val voice=VoicePrint(ctx.assets)
  try {
   val t=voice.embed(load("1089-134686-0013.wav").take(48000).toFloatArray()) ?: error("SkiM enrollment failed")
   val automatic=AutoVoiceReference();automatic.setTarget(1);val ref=load("1089-134686-0013.wav")
   repeat(3){automatic.collect(ref.copyOfRange(it*32000,(it+1)*32000),true)?.let{p->automatic.accept(voice.embed(p))}}
   check(automatic.centroid!=null);log("SKIM_AUTO_REFERENCE ready=true phrases=${automatic.phrases} scriptedVisualSolo=true noCameraValidation=true")
   for((name,mix) in cases){
    val model=SkiMStream(ctx);val down=Downsample16To8();val up=Array(2){Upsample8To16()};val low=down.process(mix);val sources=Array(2){FloatArray(mix.size)};val times=ArrayList<Long>()
    try{for(i in 0 until low.size-204 step 200){val start=SystemClock.elapsedRealtimeNanos();val y=model.process(low.copyOfRange(i,i+204));times.add((SystemClock.elapsedRealtimeNanos()-start)/1000000);for(s in 0..1)up[s].process(y[s]).copyInto(sources[s],i*2)}}finally{model.close()}
    check(sources.all{src->src.all{it.isFinite()}});val begin=4000;val end=92000
    val scores=sources.map{sdr(it.copyOfRange(begin+7,end+7),a.copyOfRange(begin,end))};val best=scores.maxOrNull()!!
    times.sort();log("SKIM_MODEL case=$name chunkMs=25 p50Ms=${times[(times.size-1)/2]} p95Ms=${times[(times.size-1)*95/100]} deadlineMisses=${times.count{it>25}}/${times.size} oracleAbsoluteSI_SDR=$best oracleSI_SDRi=${best-sdr(mix.copyOfRange(begin,end),a.copyOfRange(begin,end))} peak=${sources.maxOf{src->src.maxOf{abs(it)}}} finite=true sampleRate=8000 resamplerDelay16kSamples=7 notProductionPolicyOutput=true")
    val sep=ExperimentalSeparator(ctx,voice,{t},{.35f},{.10f},{false});val rendered=FloatArray(mix.size);var fallback=0;var warmup=0;var selected=0;var first=-1;val start=SystemClock.elapsedRealtimeNanos()
    try{for(pos in mix.indices step 160){val deadline=start+pos*1000000000L/16000;val wait=deadline-SystemClock.elapsedRealtimeNanos();if(wait>0)Thread.sleep(wait/1000000,(wait%1000000).toInt());val out=FloatArray(160);sep.process(mix.copyOfRange(pos,pos+160),out);out.copyInto(rendered,pos);if(pos>=3200){if(sep.underflowSamples>0)fallback++;if(sep.selectedOutput){selected++;if(first<0)first=pos/16}};if(sep.warmupSamples>0)warmup++}
     check(rendered.all{it.isFinite()});val lo=4000;val hi=90000;val q=sdr(rendered.copyOfRange(lo+3200,hi+3200),a.copyOfRange(lo,hi));log("SKIM_PIPELINE case=$name postWarmFrames=${(mix.size-3200)/160} rawFallbackFrames=$fallback selectedFrames=$selected firstSelectedMs=$first warmupFrames=$warmup deliveredPolicySI_SDR=$q fixedSoftwareDelayMs=200 noMic=true noCamera=true noPlayback=true noNegativeMute=true")
    }finally{sep.stop()}
   }
   log("PASS causal SKIM models and production fixtures finite. Not room/BT validated.")
   return "PASS SKIM"
  }finally{voice.release()}
 }
}
