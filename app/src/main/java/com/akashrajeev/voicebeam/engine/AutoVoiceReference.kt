package com.akashrajeev.voicebeam.engine
import com.akashrajeev.voicebeam.core.VoiceMatch
import kotlin.math.sqrt
/** Ephemeral auto reference, gated by visual solo speech. Not proof of identity. */
class AutoVoiceReference {
 @Volatile var centroid:FloatArray?=null;private set
 @Volatile var phrases=0;private set
 @Volatile var lastDecision="Waiting for solo speech";private set
 private val samples=ArrayList<Float>()
 private var previous:FloatArray?=null
 private var key:Int?=null
 @Synchronized fun capturedMillis():Int=samples.size/16
 @Synchronized fun setTarget(id:Int?){if(key!=id){key=id;centroid=null;phrases=0;previous=null;samples.clear();lastDecision="Target changed: reference reset"}}
 @Synchronized fun collect(raw:FloatArray,eligible:Boolean):FloatArray? {
  if(centroid!=null)return null
  if(!eligible){if(samples.isNotEmpty())lastDecision="Solo gate closed: partial phrase reset";samples.clear();return null}
  for(v in raw)samples.add(v)
  if(samples.size<32000)return null
  val a=samples.take(32000).toFloatArray();samples.clear()
  val rms=sqrt(a.fold(0.0){s,v->s+v*v}/a.size)
  lastDecision=if(rms>=.005)"Phrase captured: embedding pending" else "Phrase rejected: RMS below.005"
  return if(rms>=.005)a else null
 }
 @Synchronized fun accept(e:FloatArray?):Boolean {
  if(e==null || e.isEmpty() || e.any{!it.isFinite()} || e.sumOf{it.toDouble()*it}<1e-8){previous=null;phrases=0;lastDecision="Invalid embedding: phrases reset";return false}
  val prev=previous
  if(prev!=null && VoiceMatch.cosine(prev,e)<.55f){previous=e;phrases=1;lastDecision="Speaker mismatch: restarted at1/3";return false}
  phrases++
  val mean=if(prev==null)e.copyOf() else FloatArray(e.size){(prev[it]*(phrases-1)+e[it])/phrases}
  previous=mean
  if(phrases>=3)centroid=mean
  lastDecision=if(centroid!=null)"Reference ready" else "Consistent phrase accepted: $phrases/3"
  return centroid!=null
 }
}
