package com.akashrajeev.voicebeam.core

/** 16k->8k antialias FIR; causal state and fixed sample count for even chunks. */
class Downsample16To8 {
 private val taps=floatArrayOf(-.00365145f,0f,.01617925f,0f,-.06841178f,0f,.30494752f,.50187293f,.30494752f,0f,-.06841178f,0f,.01617925f,0f,-.00365145f)
 private val ring=FloatArray(15);private var pos=0;private var count=0
 fun reset(){ring.fill(0f);pos=0;count=0}
 fun process(input:FloatArray):FloatArray {
  val out=FloatArray((input.size+count%2)/2);var w=0
  for(v in input){ring[pos]=v;pos=(pos+1)%15;count++;if(count%2==0){var sum=0f;for(i in taps.indices)sum+=taps[i]*ring[(pos+14-i)%15];out[w++]=sum}}
  return out
 }
}
/** 8k->16k windowed-sinc half-sample interpolator. Original samples pass exactly; 4-sample lookahead adds 8 samples at 16k. */
class Upsample8To16 {
 private val c=DoubleArray(4){m->val x=m+0.5;val sinc=Math.sin(Math.PI*x)/(Math.PI*x);val w=0.5+0.5*Math.cos(Math.PI*x/4.0);sinc*w}.let{a->val sum=a.sum()*2;DoubleArray(4){a[it]/sum}}
 private val h=FloatArray(8)
 fun reset(){h.fill(0f)}
 fun process(input:FloatArray):FloatArray {
  val out=FloatArray(input.size*2)
  for(i in input.indices){
   for(k in 0 until 7)h[k]=h[k+1];h[7]=input[i]
   // h[0..7]=in[i-7..i]; emitted index j=i-4 is h[3], previous h[2], next h[4]
   var mid=0.0;for(m in 0..3)mid+=c[m]*(h[3-m]+h[4+m])
   out[i*2]=mid.toFloat();out[i*2+1]=h[3]
  }
  return out
 }
}
