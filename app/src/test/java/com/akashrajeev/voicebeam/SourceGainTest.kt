package com.akashrajeev.voicebeam
import com.akashrajeev.voicebeam.core.SourceGain
import org.junit.Test
import org.junit.Assert.*
class SourceGainTest {
 @Test fun invertedLargeSourceReconstructsMix(){val s=arrayOf(floatArrayOf(-2f,-4f),floatArrayOf(-2f,-4f));val g=SourceGain().apply(s,floatArrayOf(.2f,.4f));assertEquals(-.05f,g,1e-6f);assertArrayEquals(floatArrayOf(.1f,.2f),s[0],1e-6f)}
 @Test fun silenceFinite(){val s=Array(2){FloatArray(10)};assertEquals(0f,SourceGain().apply(s,FloatArray(10)),0f)}
 @Test fun peakProtected(){val s=arrayOf(floatArrayOf(20f,1f),floatArrayOf(-19f,1f));SourceGain().apply(s,floatArrayOf(1f,1f));assertTrue(s.all{v->v.all{kotlin.math.abs(it)<=.95001f}})}
}
