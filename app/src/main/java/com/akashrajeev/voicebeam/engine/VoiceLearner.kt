package com.akashrajeev.voicebeam.engine

import com.akashrajeev.voicebeam.core.VoiceMatch

/**
 * Learns the locked speaker's voice from moments when only their lips move,
 * then scores new speech against it. Pure logic; the embedding function is injected.
 */
class VoiceLearner(
    private val embed: (FloatArray) -> FloatArray?,
    private val sampleRate: Int = 16000,
    private val chunkSeconds: Float = 3f,
    private val needed: Int = 3,
) {
    private val chunk = (sampleRate * chunkSeconds).toInt()
    private val enrollBuf = FloatArray(chunk)
    private var enrollFill = 0
    private val scoreBuf = FloatArray(chunk)
    private var scoreFill = 0
    private val prints = mutableListOf<FloatArray>()
    var centroid: FloatArray? = null
        private set
    val learned: Boolean get() = centroid != null
    val progress: Float get() = if (learned) 1f else prints.size / needed.toFloat()

    /** Feed voiced audio. [lipWeight] is how sure we are the locked person is the one talking. Returns a new match score when one is ready. */
    @Synchronized
    fun feed(samples: FloatArray, lipWeight: Float): Float? {
        if (!learned) {
            if (lipWeight >= 0.75f) {
                enrollFill = append(enrollBuf, enrollFill, samples)
                if (enrollFill >= chunk) {
                    embed(enrollBuf.copyOf())?.let { prints.add(it) }
                    enrollFill = 0
                    if (prints.size >= needed) centroid = VoiceMatch.average(prints)
                }
            }
            else { enrollFill = 0 }
            return null
        }
        scoreFill = append(scoreBuf, scoreFill, samples)
        if (scoreFill < chunk) return null
        scoreFill = 0
        val e = embed(scoreBuf.copyOf()) ?: return null
        val c = centroid ?: return null
        val score = VoiceMatch.score(VoiceMatch.cosine(e, c))
        // Lab: freeze enrolment; self-confirming adaptation can learn a distractor.
        return score
    }

    private fun append(buf: FloatArray, fill: Int, s: FloatArray): Int {
        val n = minOf(s.size, buf.size - fill)
        System.arraycopy(s, 0, buf, fill, n)
        return fill + n
    }

    @Synchronized
    fun reset() { prints.clear(); centroid = null; enrollFill = 0; scoreFill = 0 }
}
