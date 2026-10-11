package com.akashrajeev.voicebeam.engine

/** Scores and waveform continuity are provisional, never proof of target presence.
 * New silence is permitted only by repeated FRESH valid negative identity results.
 * A negative expires quickly: a newly entering target must not inherit old mute.
 */
class SeparationPresence {
    enum class Decision { SELECT, RAW, MUTE }
    private var negatives=0
    private var updated=Long.MIN_VALUE
    private var selected=false
    private var negative=false
    fun reset(){negatives=0;updated=Long.MIN_VALUE;selected=false;negative=false}
    fun decide(scores:List<Float>,fresh:Boolean,nowMs:Long,templateValid:Boolean,threshold:Float=.35f,margin:Float=.10f):Decision {
        if(!templateValid){reset();return Decision.RAW}
        val valid=scores.size==2 && scores.all{it.isFinite() && it>=0f}
        if(fresh){
            updated=nowMs
            val hi=scores.maxOrNull() ?: -1f;val lo=scores.minOrNull() ?: -1f
            selected=valid && hi>threshold && hi-lo>margin
            negative=valid && hi<threshold-.10f
            negatives=if(negative)negatives+1 else 0
        }
        val age=if(updated==Long.MIN_VALUE)Long.MAX_VALUE else nowMs-updated
        if(age<0 || !valid)return Decision.RAW
        if(selected && age<=1000)return Decision.SELECT
        if(negative && negatives>=2 && age<=250)return Decision.MUTE
        return Decision.RAW
    }
}
