package com.akashrajeev.voicebeam.engine

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.akashrajeev.voicebeam.core.*
import com.akashrajeev.voicebeam.ml.VoicePrint
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/** Causal SkiM trial. Source identity remains provisional; never negative mute. */
class ExperimentalSeparator(
 app:Context, private val voice:VoicePrint, private val centroid:()->FloatArray?,
 private val threshold:()->Float, private val margin:()->Float, private val rawSelection:()->Boolean,
 contextMs:Int=1000,hopMs:Int=500,
 private val autoTarget:()->Int?={null},private val autoEligible:()->Boolean={false},
) {
 private data class Input(val audio:FloatArray,val start:Long)
 private data class Output(val audio:FloatArray,val start:Long,val template:FloatArray?,val isolated:Boolean,val generation:Int)
 private data class Identity(val sources:Array<FloatArray>?,val template:FloatArray?,val reference:FloatArray?,val target:Int?,val generation:Int=0)
 private val identityVoice=VoicePrint(app.assets)
 private val model=SkiMStream(app)
 private val down=Downsample16To8();private val up=Array(2){Upsample8To16()}
 private val renderer=IndexedSeparationRenderer(3200,16000,80)
 private val automatic=AutoVoiceReference()
 @Volatile var referenceEligible=false;private set
 fun referenceStatus():String = when {centroid()!=null -> "Manual target reference ready"; automatic.centroid!=null -> "Auto reference ready: provisional separation"; autoTarget()==null -> "Tap a target face first"; !referenceEligible -> "Auto reference ${automatic.phrases}/3: waiting for visible target speaking alone"; else -> "Auto reference ${automatic.phrases}/3: collecting ${automatic.capturedMillis()/1000f}s / 2s solo speech"}
 fun referenceReady()=template()!=null
 fun continuityStatus()="underflowSamples=$fallbackTotal fallbackSec=${fallbackTotal/16000f} warmupSamples=$warmupTotal renderedSamples=$renderedTotal lateSamples=$late inputDrops=$drops outputDrops=${outputDrops} switchCount=${renderer.switchCount} missingRawSamples=${renderer.missingRawSamples}"
 private fun template()=centroid() ?: automatic.centroid
 private val input=ArrayBlockingQueue<Input>(8);private val output=ArrayBlockingQueue<Output>(16)
 private val identity=ArrayBlockingQueue<Identity>(1)
 private val running=AtomicBoolean(true)
 @Volatile private var generation=0
 private data class Scores(val values:List<Float>,val time:Long,val template:FloatArray?,val generation:Int)
 @Volatile private var identityScores=Scores(listOf(-1f,-1f),0,null,-1)
 @Volatile var selectedOutput=false;private set
 @Volatile var underflowSamples=0;private set
 @Volatile var warmupSamples=0;private set
 private var fallbackTotal=0L;private var warmupTotal=0L;private var renderedTotal=0L
 private var referenceLogAt=0L
 private var renderedGeneration=-1;private var captured=0L;private var rawTarget:Int?=null;private var renderedTemplate:FloatArray?=null
 private val buffer=FloatArray(204);private var fill=0;private var blockStart=0L
 @Volatile private var outputDrops=0
 private var drops=0;private var late=0
 private val worker:Thread
 private val idWorker:Thread
 init {
  Diagnostics.event("SKIM causal=true chunkMs=25 fixedDelayMs=200 sampleRate=8000 model=int8 failOpenNoMute=true")
  idWorker=Thread({identityLoop()},"vb-skim-id").also{it.start()}
  worker=Thread({loop()},"vb-skim").also{it.start()}
 }
 fun process(raw:FloatArray,out:FloatArray):Boolean {
  val target=autoTarget();if(target!=rawTarget){automatic.setTarget(target);rawTarget=target;identityScores=Scores(listOf(-1f,-1f),0,null,-1)}
  referenceEligible=autoEligible()
  if(centroid()==null && automatic.centroid==null)automatic.collect(raw,referenceEligible)?.let{identity.offer(Identity(null,null,it,target))}
  val now=SystemClock.elapsedRealtime()
  if(now-referenceLogAt>=1000){referenceLogAt=now;Diagnostics.event("AUTO_REFERENCE phrases=${automatic.phrases} capturedMs=${automatic.capturedMillis()} eligible=$referenceEligible ready=${referenceReady()} last=${automatic.lastDecision}")}
  renderer.push(raw)
  for(v in down.process(raw)){
   buffer[fill++]=v
   if(fill==204){val p=Input(buffer.copyOf(),blockStart);if(!input.offer(p))drops++;buffer.copyInto(buffer,0,200,204);fill=4;blockStart+=400}
  }
  captured+=raw.size
  val t=template();if(t !== renderedTemplate || renderedGeneration!=generation){renderer.clearSeparated();renderedTemplate=t;renderedGeneration=generation}
  while(true){val p=output.poll() ?: break;if(p.template!==t || p.generation!=generation){late++;continue};late+=renderer.offer(p.start,p.audio,p.isolated)}
  val f=renderer.render(out);selectedOutput=f.isolated;underflowSamples=f.rawSamples;warmupSamples=f.warmupSamples
  fallbackTotal+=f.rawSamples;warmupTotal+=f.warmupSamples;renderedTotal+=out.size
  return f.rawSamples+f.warmupSamples<out.size
 }
 private fun identityLoop(){
  try{while(running.get()){
   val p=identity.poll(200,java.util.concurrent.TimeUnit.MILLISECONDS) ?: continue
   if(p.reference!=null){val e=identityVoice.embed(p.reference);synchronized(automatic){if(p.target==autoTarget())automatic.accept(e)};continue}
   if(p.template==null || p.template!==template() || p.generation!=generation)continue
   val s=p.sources!!.map{identityVoice.embed(it)?.let{e->VoiceMatch.cosine(e,p.template)} ?: -1f}
   if(p.template===template() && p.generation==generation){identityScores=Scores(s,SystemClock.elapsedRealtime(),p.template,p.generation)}
  }}catch(t:Throwable){Diagnostics.event("SKIM identityError="+t.javaClass.simpleName)}finally{identityVoice.release()}
 }
 private fun loop(){var expected=0L;var idFill=0;val idAudio=Array(2){FloatArray(16000)};var idTemplate:FloatArray?=null;val times=ArrayList<Long>();var n=0
  try{while(running.get()){
   val p=input.poll(200,java.util.concurrent.TimeUnit.MILLISECONDS) ?: continue
   if(p.start!=expected){generation++;model.reset();up.forEach{it.reset()};idFill=0;identityScores=Scores(listOf(-1f,-1f),0,null,-1);Diagnostics.event("SKIM stateReset gapSamples="+(p.start-expected))};expected=p.start+400
   val start=SystemClock.elapsedRealtimeNanos();val src=model.process(p.audio);val samples=Array(2){up[it].process(src[it])};val t=template()
   if(t!==idTemplate){generation++;idFill=0;idTemplate=t;identityScores=Scores(listOf(-1f,-1f),0,null,-1)}
   for(s in 0..1)samples[s].copyInto(idAudio[s],idFill);idFill+=400
   if(idFill==16000){identity.offer(Identity(Array(2){idAudio[it].copyOf()},t,null,null,generation));idFill=0}
   val result=identityScores;val s=result.values;val best=if(s[0]>=s[1])0 else 1
   val fresh=result.template===t && result.generation==generation && SystemClock.elapsedRealtime()-result.time<=1000
   val accepted=t!=null && fresh && s[best]>threshold() && s[best]-s[1-best]>margin()
   val isolated=accepted && !rawSelection()
   // No uncertain output packet: indexed renderer retains the same-time raw fallback.
   if(isolated || rawSelection())if(!output.offer(Output(samples[best],p.start-7,t,isolated,generation)))outputDrops++
   val ms=(SystemClock.elapsedRealtimeNanos()-start)/1000000;times.add(ms);n++
   Diagnostics.tseState=if(accepted)"Provisional causal target" else "UNCERTAIN: delayed raw"
   if(n%40==0){val sorted=times.sorted();val row="SKIM chunks=$n p50Ms=${sorted[(sorted.size-1)/2]} p95Ms=${sorted[(sorted.size-1)*95/100]} dropsInput=$drops lateSamples=$late backlog=${input.size} fixedDelayMs=200 noNegativeMute=true ${continuityStatus()}";Diagnostics.event(row);Log.i("VoiceBeamSelfTest",row)}
  }}catch(t:Throwable){Diagnostics.tseState="SKIM ERROR "+t.javaClass.simpleName;Diagnostics.event(Diagnostics.tseState);Log.e("VoiceBeamSelfTest","SKIM error",t)}finally{model.close()}
 }
 fun stop(){running.set(false);worker.join(5000);idWorker.join(5000);input.clear();output.clear();identity.clear();renderer.clearSeparated();Diagnostics.tseState="OFF"}
}
