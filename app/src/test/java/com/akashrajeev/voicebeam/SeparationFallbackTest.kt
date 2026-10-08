package com.akashrajeev.voicebeam
import com.akashrajeev.voicebeam.core.SeparationFallback
import org.junit.Test
import org.junit.Assert.*
class SeparationFallbackTest {
 @Test fun fullUnderflowPassesRawWithoutBoost(){val out=FloatArray(3);assertEquals(3,SeparationFallback.fill(floatArrayOf(.2f,-.3f,1f),out,0));assertArrayEquals(floatArrayOf(.2f,-.3f,.95f),out,0f)}
 @Test fun partialUnderflowPreservesAvailablePrefix(){val out=floatArrayOf(.7f,0f,0f);assertEquals(2,SeparationFallback.fill(floatArrayOf(.1f,.2f,.3f),out,1));assertArrayEquals(floatArrayOf(.7f,.2f,.3f),out,0f)}
 @Test fun fullPacketIsUnchanged(){val out=floatArrayOf(.7f);assertEquals(0,SeparationFallback.fill(floatArrayOf(.1f),out,1));assertArrayEquals(floatArrayOf(.7f),out,0f)}
}
