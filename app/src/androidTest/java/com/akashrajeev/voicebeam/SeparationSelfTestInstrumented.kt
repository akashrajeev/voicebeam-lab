package com.akashrajeev.voicebeam
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.akashrajeev.voicebeam.engine.SeparationSelfTest
import org.junit.Test
import org.junit.Assert.assertTrue
import org.junit.runner.RunWith
@RunWith(AndroidJUnit4::class)
class SeparationSelfTestInstrumented {
    @Test fun publicFixtureNativeLoop() {
        com.akashrajeev.voicebeam.engine.SeparationPipelineProbe.run(InstrumentationRegistry.getInstrumentation().targetContext)
        val result=SeparationSelfTest.run(InstrumentationRegistry.getInstrumentation().targetContext)
        assertTrue(result.contains("SI_SDRi="));assertTrue(result.contains("PASS native models"))
    }
}
