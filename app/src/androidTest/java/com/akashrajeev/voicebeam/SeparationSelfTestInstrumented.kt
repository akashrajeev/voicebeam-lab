package com.akashrajeev.voicebeam
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.akashrajeev.voicebeam.engine.SkiMSelfTest
import org.junit.Test
import org.junit.Assert.assertEquals
import org.junit.runner.RunWith
@RunWith(AndroidJUnit4::class)
class SeparationSelfTestInstrumented {
 @Test fun publicFixtureNativeLoop(){assertEquals("PASS SKIM",SkiMSelfTest.run(InstrumentationRegistry.getInstrumentation().targetContext))}
}
