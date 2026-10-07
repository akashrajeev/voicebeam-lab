package com.akashrajeev.voicebeam

import com.akashrajeev.voicebeam.core.*
import org.junit.Assert.*
import org.junit.Test

class SuppressionTest {
    private val target = GateInputs(true, 0.9f, 0f, null, true)
    private val self = GateInputs(true, 0f, 0f, null, true)

    @Test fun staleTargetProbabilityExpiresAfterVadSilence() {
        val g = TargetGate(frameMs = 10f)
        repeat(100) { g.process(target) }
        repeat(150) { g.process(self.copy(voiceActive = false)) }
        assertEquals(0f, g.probability, 0f)
        assertFalse(g.boostAllowed)
        assertTrue(g.gain < 0.041f)
    }
    @Test fun offCameraSelfVoiceGetsNoBoostAndStrongAttenuation() {
        val g = TargetGate(frameMs = 10f)
        repeat(100) { g.process(target) }
        repeat(100) { g.process(self) }
        assertEquals(TargetState.UNCERTAIN, g.state)
        assertFalse(g.boostAllowed)
        assertTrue(g.gain < 0.041f)
    }
    @Test fun freshNegativeSpeakerMatchVetoesLipMotion() {
        val g = TargetGate()
        assertEquals(0f, g.targetProbability(target.copy(voiceMatch = 0.1f)), 0f)
        assertEquals(TargetState.OTHER, g.state)
    }
    @Test fun fullyQuietOthersConvergesToMuteNotBoostedResidual() {
        val g = TargetGate(frameMs = 10f); g.quietOthers = 1f
        repeat(100) { g.process(target) }
        repeat(150) { g.process(self) }
        assertTrue(g.gain < 0.000001f); assertFalse(g.boostAllowed)
    }
    @Test fun confirmedTargetStillPassesAndCanBeBoosted() {
        val g = TargetGate(frameMs = 10f)
        repeat(100) { g.process(target) }
        assertTrue(g.gain > 0.95f); assertTrue(g.boostAllowed)
    }
    @Test fun newlyLockedQuietFaceDoesNotInheritOpenMonitor() {
        val g = TargetGate(); repeat(50) { g.process(self.copy(hasLock = false)) }
        assertTrue(g.process(self) < 0.05f); assertFalse(g.boostAllowed)
    }
    @Test fun overlapIsNotGivenTargetBoost() {
        val g = TargetGate(); g.process(target.copy(othersSpeaking = 0.9f))
        assertEquals(TargetState.OVERLAP, g.state); assertFalse(g.boostAllowed)
    }
    @Test fun zeroSuppressionStillHonorsUserSlider() {
        val g = TargetGate(frameMs = 10f); g.quietOthers = 0f
        repeat(100) { g.process(self) }
        assertTrue(g.gain > 0.99f); assertFalse(g.boostAllowed)
    }
}
