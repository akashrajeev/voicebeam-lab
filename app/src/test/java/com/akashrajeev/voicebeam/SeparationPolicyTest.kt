package com.akashrajeev.voicebeam
import com.akashrajeev.voicebeam.engine.SeparationPresence
import com.akashrajeev.voicebeam.engine.SlowIdentity
import org.junit.Test
import org.junit.Assert.*
class SeparationPolicyTest {
    @Test fun noTemplateOrFailedScoresAlwaysPassRaw(){
        val p=SeparationPresence()
        assertEquals(SeparationPresence.Decision.RAW,p.decide(listOf(.9f,.1f),true,0,false))
        assertEquals(SeparationPresence.Decision.RAW,p.decide(listOf(Float.NaN,.1f),true,0,true))
        assertEquals(SeparationPresence.Decision.RAW,p.decide(listOf(-1f,-1f),true,0,true))
    }
    @Test fun negativeNeedsTwoFreshResultsAndExpires(){
        val p=SeparationPresence();val s=listOf(.1f,.12f)
        assertEquals(SeparationPresence.Decision.RAW,p.decide(s,true,0,true))
        assertEquals(SeparationPresence.Decision.RAW,p.decide(s,false,100,true))
        assertEquals(SeparationPresence.Decision.MUTE,p.decide(s,true,1000,true))
        assertEquals(SeparationPresence.Decision.RAW,p.decide(s,false,1251,true))
        assertEquals(SeparationPresence.Decision.SELECT,p.decide(listOf(.8f,.1f),true,1300,true))
    }
    @Test fun uncertainAndTemplateResetNeverInheritMute(){
        val p=SeparationPresence();p.decide(listOf(.1f,.1f),true,0,true);p.decide(listOf(.1f,.1f),true,1000,true)
        assertEquals(SeparationPresence.Decision.RAW,p.decide(listOf(.4f,.4f),true,1100,true))
        assertEquals(SeparationPresence.Decision.RAW,p.decide(listOf(.1f,.1f),true,1200,false))
        assertEquals(SeparationPresence.Decision.RAW,p.decide(listOf(.1f,.1f),true,1300,true))
    }
    @Test fun identityRunsOncePer16000NonDuplicatedSamples(){
        val id=SlowIdentity(4000,16000);var calls=0
        repeat(3){assertFalse(id.update(arrayOf(FloatArray(16000){.1f},FloatArray(16000){.2f})){calls++;.8f}.fresh)}
        assertEquals(0,calls)
        assertTrue(id.update(arrayOf(FloatArray(16000){.1f},FloatArray(16000){.2f})){calls++;.8f}.fresh)
        assertEquals(2,calls)
    }
}
