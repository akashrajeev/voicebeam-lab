package com.akashrajeev.voicebeam.engine
import com.akashrajeev.voicebeam.core.VoiceMatch
import kotlin.math.sqrt
/** Ephemeral auto reference, gated by visual solo speech. Not proof of identity. */
class AutoVoiceReference {
 @Volatile var centroid:FloatArray?=null;private set
 @Volatile var phrases=0;private set
 private val samples=ArrayList<Float>()
 private var previous:FloatArray?=null
 private var key:Int?=null
 @Synchronized fun setTarget(id:Int?){if(key!=id){key=id;centroid=null;phrases=0;previous=null;samples.clear()}}
 @Synchronized fun collect(raw:FloatArray,eligible:Boolean):FloatArray? {
  if(centroid!=null)return null
  if(!eligible){samples.clear();return null}
  for(v in raw)samples.add(v)
  if(samples.size<32000)return null
  val a=samples.take(32000).toFloatArray();samples.clear()
  val rms=sqrt(a.fold(0.0){s,v->s+v*v}/a.size)
  return if(rms>=.005)a else null
 }
 @Synchronized fun accept(e:FloatArray?):Boolean {
  if(e==null || e.any{!it.isFinite()}){previous=null;phrases=0;return false}
  val prev=previous
  if(prev!=null && VoiceMatch.cosine(prev,e)<.55f){previous=e;phrases=1;return false}
  phrases++
  val mean=if(prev==null)e.copyOf() else FloatArray(e.size){(prev[it]*(phrases-1)+e[it])/phrases}
  previous=mean
  if(phrases>=3)centroid=mean
  return centroid!=null
 }
}
