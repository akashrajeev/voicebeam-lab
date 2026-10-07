package com.akashrajeev.voicebeam.engine

import kotlin.math.sqrt

/** Identity at 1Hz from non-duplicated selected hops. Continuity is not presence. */
class SlowIdentity(private val hop:Int, private val context:Int) {
    data class Result(val scores:List<Float>,val fresh:Boolean)
    private var prev:Array<FloatArray>?=null
    private val chunks= arrayOf(ArrayList<Float>(),ArrayList<Float>())
    private var cached=listOf(-1f,-1f)
    fun update(src:Array<FloatArray>,shift:Int=hop,embed:(FloatArray)->Float):Result {
        val p=prev
        var order=intArrayOf(0,1)
        if(p!=null && shift in 1 until context) {
            fun corr(a:FloatArray,b:FloatArray):Double {
                var d=0.0;var aa=0.0;var bb=0.0
                for(k in 0 until context-shift){val x=a[k+shift].toDouble();val y=b[k].toDouble();d+=x*y;aa+=x*x;bb+=y*y}
                return d/(sqrt(aa*bb)+1e-8)
            }
            if(corr(p[0],src[1])+corr(p[1],src[0])>corr(p[0],src[0])+corr(p[1],src[1]))order=intArrayOf(1,0)
        } else if(p!=null) {chunks.forEach{it.clear()};cached=listOf(-1f,-1f)}
        prev=arrayOf(src[order[0]].copyOf(),src[order[1]].copyOf())
        for(c in 0..1)for(k in context-hop until context) chunks[c].add(src[order[c]][k])
        val fresh=chunks[0].size>=16000
        if(fresh){cached=(0..1).map{c->embed(chunks[c].takeLast(16000).toFloatArray())};chunks.forEach{it.clear()}}
        val scores=MutableList(2){-1f};for(c in 0..1)scores[order[c]]=cached[c]
        return Result(scores,fresh)
    }
}
