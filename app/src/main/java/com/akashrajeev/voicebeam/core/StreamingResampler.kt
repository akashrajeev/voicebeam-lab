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
/** Linear interpolation8k->16k with carried previous sample, causal1sample delay. */
class Upsample8To16 {
 private var prev=0f
 fun reset(){prev=0f}
 fun process(input:FloatArray):FloatArray {val out=FloatArray(input.size*2);for(i in input.indices){out[i*2]=(prev+input[i])*.5f;out[i*2+1]=input[i];prev=input[i]};return out}
}
