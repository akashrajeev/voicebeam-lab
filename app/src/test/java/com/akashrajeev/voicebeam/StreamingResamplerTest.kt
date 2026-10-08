package com.akashrajeev.voicebeam
import com.akashrajeev.voicebeam.core.*
import org.junit.Test
import org.junit.Assert.*
class StreamingResamplerTest {
 @Test fun chunkedEqualsWhole(){val x=FloatArray(800){kotlin.math.sin(it*.2).toFloat()};val all=Downsample16To8().process(x);val d=Downsample16To8();val parts=d.process(x.copyOfRange(0,400))+d.process(x.copyOfRange(400,800));assertArrayEquals(all,parts,0f)
  val u=Upsample8To16();val p=u.process(all.copyOfRange(0,200))+u.process(all.copyOfRange(200,400));assertArrayEquals(Upsample8To16().process(all),p,0f)}
 @Test fun constantSettlesAndReset(){val d=Downsample16To8();val a=d.process(FloatArray(400){.2f});assertEquals(.2f,a.last(),.001f);d.reset();assertArrayEquals(a,d.process(FloatArray(400){.2f}),0f)}
}
