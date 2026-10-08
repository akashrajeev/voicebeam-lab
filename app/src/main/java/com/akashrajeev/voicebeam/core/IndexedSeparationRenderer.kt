package com.akashrajeev.voicebeam.core

/** Audio-thread-owned fixed-delay schedule. Late separated samples never replay. */
class IndexedSeparationRenderer(val delaySamples:Int, capacity:Int=delaySamples*3, private val seamSamples:Int=0) {
    data class Frame(val isolated:Boolean,val rawSamples:Int,val warmupSamples:Int)
    private val raw=FloatArray(capacity)
    private val extracted=FloatArray(capacity)
    private val rawIndex=LongArray(capacity){-1L}
    private val sepIndex=LongArray(capacity){-1L}
    private val isolated=BooleanArray(capacity)
    private var captured=0L
    private var rendered=0L
    private var blend=0f
    private var residual=0f
    private var wasReady=false
    var switchCount=0L;private set
    var missingRawSamples=0L;private set
    init { require(delaySamples>0 && capacity>delaySamples) }
    private fun slot(i:Long)=(i%raw.size).toInt()
    fun clearSeparated(){sepIndex.fill(-1L)}
    fun push(input:FloatArray){for(v in input){val k=slot(captured);raw[k]=v.coerceIn(-.95f,.95f);rawIndex[k]=captured;captured++}}
    fun offer(start:Long,audio:FloatArray,selected:Boolean):Int {
        var late=0
        for(k in audio.indices){val i=start+k;if(i<rendered || i<0){late++;continue};if(i>=captured)continue
            val j=slot(i);if(rawIndex[j]!=i)continue
            extracted[j]=audio[k].coerceIn(-.95f,.95f);sepIndex[j]=i;isolated[j]=selected
        }
        return late
    }
    fun render(out:FloatArray):Frame {
        var rawCount=0;var warmup=0;var allSelected=true
        // Each call corresponds to this input frame, including startup silence.
        val start=captured-out.size-delaySamples
        for(k in out.indices){val i=start+k
            if(i<0){out[k]=0f;warmup++;allSelected=false;continue}
            val j=slot(i)
            val ready=sepIndex[j]==i
            if(ready!=wasReady){switchCount++;wasReady=ready}
            if(rawIndex[j]!=i)missingRawSamples++
            val rv=if(rawIndex[j]==i)raw[j] else 0f
            if(ready){blend=if(seamSamples>0)minOf(1f,blend+1f/seamSamples) else 1f;out[k]=rv*(1f-blend)+extracted[j]*blend;residual=out[k]-rv;if(!isolated[j] || blend<1f)allSelected=false}
            else {if(seamSamples>0){blend=maxOf(0f,blend-1f/seamSamples);residual*=if(blend>0f)blend/(blend+1f/seamSamples) else 0f}else{blend=0f;residual=0f};out[k]=(rv+residual).coerceIn(-.95f,.95f);rawCount++;allSelected=false}
            rendered=i+1
        }
        return Frame(allSelected,rawCount,warmup)
    }
}
