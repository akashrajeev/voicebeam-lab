package com.akashrajeev.voicebeam
import com.akashrajeev.voicebeam.engine.AutoVoiceReference
import org.junit.Test
import org.junit.Assert.*
class AutoReferenceTest {
 @Test fun requiresSoloSpeechAndThreeConsistentPhrases(){val a=AutoVoiceReference();a.setTarget(1)
  assertNull(a.collect(FloatArray(32000){.02f},false));assertNull(a.collect(FloatArray(32000),true))
  repeat(2){assertNotNull(a.collect(FloatArray(32000){.02f},true));assertFalse(a.accept(floatArrayOf(1f,0f)))}
  assertTrue(a.accept(floatArrayOf(1f,0f)));assertNotNull(a.centroid)
  a.setTarget(2);assertNull(a.centroid);assertEquals(0,a.phrases)
 }
 @Test fun conflictingVoiceCannotCommit(){val a=AutoVoiceReference();a.setTarget(1);a.accept(floatArrayOf(1f,0f));assertFalse(a.accept(floatArrayOf(0f,1f)));assertNull(a.centroid)}
}
