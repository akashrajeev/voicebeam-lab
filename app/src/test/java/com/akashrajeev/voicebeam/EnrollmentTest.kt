package com.akashrajeev.voicebeam

import com.akashrajeev.voicebeam.engine.VoiceLearner
import org.junit.Assert.*
import org.junit.Test

class EnrollmentTest {
    @Test fun implicitSpeechDoesNotCreateVoiceTemplate() {
        val l = VoiceLearner({ floatArrayOf(1f, 0f) }, 1000, 1f, 3)
        repeat(50) { l.feed(FloatArray(250){.02f}, 1f) }
        assertFalse(l.learned); assertEquals(0f, l.progress, 0f)
    }
    @Test fun explicitProgressAndFrozenCompletion() {
        val l = VoiceLearner({ floatArrayOf(1f, 0f) }, 1000, 1f, 3)
        l.beginEnrollment()
        repeat(2) { l.feed(FloatArray(250){.02f}, 1f) }
        assertEquals(1f/6f, l.progress, 0.001f)
        repeat(10) { l.feed(FloatArray(250){.02f}, 1f) }
        assertTrue(l.learned); assertFalse(l.enrollmentEnabled)
        assertEquals(1f, l.progress, 0f)
        val original = l.centroid
        repeat(50) { l.feed(FloatArray(250){.02f}, 1f) }
        assertSame(original, l.centroid)
    }
    @Test fun deliberateEnrollmentSurvivesLipDip() {
        val l = VoiceLearner({ floatArrayOf(1f, 0f) }, 1000, 1f, 3)
        l.beginEnrollment(); l.feed(FloatArray(500){.02f}, 1f)
        l.feed(FloatArray(250){.02f}, 0.1f)
        assertEquals(.25f, l.progress, .001f)
        l.beginEnrollment(); assertTrue(l.enrollmentEnabled); assertFalse(l.learned)
        l.reset(); assertFalse(l.enrollmentEnabled)
    }
    @Test fun silenceDoesNotCreateTemplate() {
        val l=VoiceLearner({floatArrayOf(1f,0f)},1000,1f,3)
        l.beginEnrollment();repeat(24){l.feed(FloatArray(250),1f)}
        assertFalse(l.learned);assertEquals(0,l.completedPhrases)
    }
}
