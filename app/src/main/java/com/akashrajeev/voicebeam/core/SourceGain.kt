package com.akashrajeev.voicebeam.core

/** Match two-source reconstruction to the same-time mixture before peak safety. */
class SourceGain {
 private var previous:Float?=null
 fun reset(){previous=null}
 fun apply(sources:Array<FloatArray>,mix:FloatArray):Float {
  var dot=0.0;var energy=0.0
  for(k in mix.indices){val v=(sources[0][k]+sources[1][k]).toDouble();dot+=v*mix[k];energy+=v*v}
  val measured=if(energy>1e-10)(dot/energy).toFloat() else 0f
  val old=previous
  val gain=if(old==null || old*measured<=0f)measured else .8f*old+.2f*measured
  previous=gain
  var peak=0f
  for(s in sources)for(k in s.indices){s[k]*=gain;peak=maxOf(peak,kotlin.math.abs(s[k]))}
  val scale=if(peak>.95f).95f/peak else 1f
  if(scale<1f)for(s in sources)for(k in s.indices)s[k]*=scale
  return gain*scale
 }
}
