package com.akashrajeev.voicebeam.engine
import android.content.Context
import com.akashrajeev.voicebeam.core.SourceGain
import ai.onnxruntime.*
import java.io.File
import java.nio.FloatBuffer

/** Pretrained8k causal SkiM,25ms block,14 explicit recurrent states. */
class SkiMStream(ctx:Context) {
 private val env=OrtEnvironment.getEnvironment()
 private val opts=OrtSession.SessionOptions().apply{setIntraOpNumThreads(2);setInterOpNumThreads(1);setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)}
 private val session:OrtSession
 private var states=Array(14){FloatArray(384)};private var active=false
 private val gain=SourceGain()
 private val tail=Array(2){FloatArray(4)}
 val inputNames:Set<String>
 init {val f=File(ctx.cacheDir,"skim-chunk-int8.onnx");if(!f.exists())ctx.assets.open("models/skim-chunk-int8.onnx").use{src->f.outputStream().use{src.copyTo(it)}};session=env.createSession(f.absolutePath,opts);inputNames=session.inputNames}
 fun reset(){states.forEach{it.fill(0f)};tail.forEach{it.fill(0f)};active=false;gain.reset()}
 fun process(audio:FloatArray):Array<FloatArray>{require(audio.size==204)
  val inputs=HashMap<String,OnnxTensor>()
  try{
   inputs["audio"]=OnnxTensor.createTensor(env,FloatBuffer.wrap(audio),longArrayOf(1,204))
   inputs["active"]=OnnxTensor.createTensor(env,FloatBuffer.wrap(floatArrayOf(if(active)1f else 0f)),longArrayOf())
   for(i in states.indices)if("state_$i" in inputNames)inputs["state_$i"]=OnnxTensor.createTensor(env,FloatBuffer.wrap(states[i]),longArrayOf(1,1,384))
   session.run(inputs).use{r->
    @Suppress("UNCHECKED_CAST") val y=(r[0].value as Array<Array<FloatArray>>)[0]
    for(i in states.indices){@Suppress("UNCHECKED_CAST") val st=r[i+1].value as Array<Array<FloatArray>>;states[i]=st[0][0].copyOf()}
    active=true
    val sources=Array(2){s->val out=y[s].copyOfRange(0,200);for(k in 0..3)out[k]+=tail[s][k];tail[s]=y[s].copyOfRange(200,204);out}
    gain.apply(sources,audio.copyOfRange(0,200));return sources
   }
  }finally{inputs.values.forEach{it.close()}}
 }
 fun close(){session.close();opts.close()}
}
