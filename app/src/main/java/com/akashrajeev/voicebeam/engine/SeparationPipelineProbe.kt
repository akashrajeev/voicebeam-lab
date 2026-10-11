package com.akashrajeev.voicebeam.engine
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.akashrajeev.voicebeam.ml.VoicePrint
import com.akashrajeev.voicebeam.core.WavWriter
import java.io.File
import kotlin.math.sqrt

/** Real paced software path. No microphone/playback. Queue gaps != Bluetooth latency. */
object SeparationPipelineProbe {
 fun run(ctx:Context) {
  fun load(name:String):FloatArray {val f=File(ctx.cacheDir,name);ctx.assets.open("selftest/$name").use{a->f.outputStream().use{a.copyTo(it)}};return WavWriter.read(f).first}
  val voice=VoicePrint(ctx.assets)
  try {
   val template=voice.embed(load("1089-134686-0013.wav").take(48000).toFloatArray()) ?: error("Probe enrollment failed")
   fun repeat(x:FloatArray):FloatArray {val rms=sqrt(x.fold(0.0){s,v->s+v*v}/x.size);return FloatArray(96000){x[it%x.size]*(.06/rms).toFloat()}}
   val a=repeat(load("1089-134686-0002.wav"));val b=repeat(load("1221-135767-0005.wav"));val mix=FloatArray(a.size){a[it]+b[it]}
   val automatic=AutoVoiceReference();automatic.setTarget(1)
   var autoReady=false
   for(i in 0..2){val phrase=load("1089-134686-0013.wav").copyOfRange(i*32000,(i+1)*32000);val captured=automatic.collect(phrase,true);autoReady=automatic.accept(captured?.let{voice.embed(it)})}
   check(autoReady){"AUTO_REFERENCE fixture did not converge"}
   Log.i("VoiceBeamSelfTest","AUTO_REFERENCE scriptedVisualSolo=true ready=$autoReady phrases=${automatic.phrases} noCameraValidation=true")
   for(context in listOf(1000,2000)) {
    val sep=ExperimentalSeparator(ctx,voice,{template},{.35f},{.10f},{false},context,500)
    var gaps=0;var fallbackFrames=0;var warmupFrames=0;var frames=0;var late=0;val start=SystemClock.elapsedRealtimeNanos()
    try {
     for(pos in mix.indices step 160) {
      val deadline=start+pos*1000000000L/16000
      val remaining=deadline-SystemClock.elapsedRealtimeNanos();if(remaining>0)Thread.sleep(remaining/1000000,(remaining%1000000).toInt()) else if(-remaining>10000000)late++
      val out=FloatArray(160);val got=sep.process(mix.copyOfRange(pos,pos+160),out)
      if(pos>=context*16){frames++;if(!got)gaps++;if(sep.underflowSamples>0)fallbackFrames++;if(sep.warmupSamples>0)warmupFrames++}
     }
     Log.i("VoiceBeamSelfTest","PIPELINE contextMs=$context hopMs=500 postWarmFrames=$frames missingOutputFrames=$gaps rawUnderflowFrames=$fallbackFrames warmupFrames=$warmupFrames fixedDelayMs=1000 pacingLateFrames=$late noMic=true noPlayback=true")
    }finally{sep.stop(); Log.i("VoiceBeamSelfTest","PRODUCTION_DIAGNOSTICS contextMs=$context\n"+Diagnostics.snapshot())}
   }
  }finally{voice.release()}
 }
}
