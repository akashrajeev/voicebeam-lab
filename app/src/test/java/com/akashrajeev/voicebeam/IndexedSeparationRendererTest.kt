package com.akashrajeev.voicebeam
import com.akashrajeev.voicebeam.core.IndexedSeparationRenderer
import org.junit.Assert.*
import org.junit.Test
class IndexedSeparationRendererTest {
 @Test fun exactDelayAndRawOrdering(){val r=IndexedSeparationRenderer(4,16);val out=FloatArray(2)
  r.push(floatArrayOf(.1f,.2f));assertEquals(2,r.render(out).warmupSamples)
  r.push(floatArrayOf(.3f,.4f));assertEquals(2,r.render(out).warmupSamples)
  r.push(floatArrayOf(.5f,.6f));assertEquals(2,r.render(out).rawSamples);assertArrayEquals(floatArrayOf(.1f,.2f),out,0f)
  r.push(floatArrayOf(.7f,.8f));r.render(out);assertArrayEquals(floatArrayOf(.3f,.4f),out,0f)
 }
 @Test fun readyPacketAndLatePacketDoNotReplay(){val r=IndexedSeparationRenderer(2,12);val out=FloatArray(2)
  r.push(floatArrayOf(.1f,.2f));r.render(out);r.offer(0,floatArrayOf(.8f,.9f),true)
  r.push(floatArrayOf(.3f,.4f));assertTrue(r.render(out).isolated);assertArrayEquals(floatArrayOf(.8f,.9f),out,0f)
  assertEquals(2,r.offer(0,floatArrayOf(.5f,.6f),true));r.push(floatArrayOf(.7f,.8f));assertFalse(r.render(out).isolated);assertArrayEquals(floatArrayOf(.3f,.4f),out,0f)
 }
 @Test fun partialPacketAndResetCannotBoostRaw(){val r=IndexedSeparationRenderer(2,12);val out=FloatArray(2)
  r.push(floatArrayOf(.1f,.2f));r.render(out);r.offer(0,floatArrayOf(.8f),true);r.push(floatArrayOf(.3f,.4f));assertFalse(r.render(out).isolated);assertArrayEquals(floatArrayOf(.8f,.2f),out,0f)
  r.offer(2,floatArrayOf(.8f,.9f),true);r.clearSeparated();r.push(floatArrayOf(.5f,.6f));assertEquals(2,r.render(out).rawSamples);assertArrayEquals(floatArrayOf(.3f,.4f),out,0f)
 }
 @Test fun ringWrapKeepsEveryRawSampleOnce(){val r=IndexedSeparationRenderer(2,6);val out=FloatArray(1)
  repeat(100){r.push(floatArrayOf(it/200f));r.render(out);if(it>=2)assertEquals((it-2)/200f,out[0],0f)}
 }
 @Test fun sameIndexSeamCannotBoostRaw(){val r=IndexedSeparationRenderer(2,12,2);val out=FloatArray(2);r.push(floatArrayOf(.2f,.2f));r.render(out);r.offer(0,floatArrayOf(.6f,.6f),true);r.push(floatArrayOf(.1f,.1f));assertFalse(r.render(out).isolated);assertArrayEquals(floatArrayOf(.4f,.6f),out,.00001f)}
}
