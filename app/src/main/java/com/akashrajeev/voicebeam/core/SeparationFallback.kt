package com.akashrajeev.voicebeam.core

/** Queue underflow passes raw audio, never silence or boosted uncertain audio. */
object SeparationFallback {
    fun fill(raw: FloatArray, out: FloatArray, written: Int): Int {
        require(written in 0..out.size)
        for (k in written until out.size) out[k] = raw.getOrElse(k) { 0f }.coerceIn(-.95f, .95f)
        return out.size - written
    }
}
