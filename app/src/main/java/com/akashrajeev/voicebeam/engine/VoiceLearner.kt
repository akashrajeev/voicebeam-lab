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
    @Volatile var lastQueryEmbedding: FloatArray? = null
        private set
    @Volatile var enrollmentEnabled = false
        private set
    @Volatile var centroid: FloatArray? = null
        private set
    @Volatile var enrollmentMessage: String = "Voice not learned"
        private set
    val learned: Boolean get() = centroid != null
    val progress: Float get() = if (learned) 1f else (prints.size + enrollFill / chunk.toFloat()) / needed.toFloat()
    val completedPhrases: Int get() = prints.size

    /** Deliberate clean target enrollment; retain frozen template until explicitly restarted. */
    @Synchronized fun beginEnrollment() {
        reset(); enrollmentEnabled = true; enrollmentMessage="Learning: only this person speaks"
    }

    /** Feed voiced audio. [lipWeight] is how sure we are the locked person is the one talking. Returns a new match score when one is ready. */
    @Synchronized
    fun feed(samples: FloatArray, lipWeight: Float): Float? {
        if (!learned) {
            if (!enrollmentEnabled) return null
            if (enrollmentEnabled) {
                enrollFill = append(enrollBuf, enrollFill, samples)
                if (enrollFill >= chunk) {
                    var energy=0.0; for(v in enrollBuf) energy+=v*v
                    if(kotlin.math.sqrt(energy/enrollBuf.size)>=.005) {
                        val embedding=embed(enrollBuf.copyOf())
                        if(embedding!=null) {prints.add(embedding);enrollmentMessage="Captured ${prints.size}/$needed phrases"}
                        else enrollmentMessage="Embedding failed: try again"
                    } else enrollmentMessage="Too quiet: move closer and speak alone"
                    enrollFill = 0
                    if (prints.size >= needed) { centroid = VoiceMatch.average(prints); enrollmentEnabled = false; enrollmentMessage="LEARNED" }
                }
            }
            else { enrollFill = 0 }
            return null
        }
        scoreFill = append(scoreBuf, scoreFill, samples)
        if (scoreFill < chunk) return null
        scoreFill = 0
        val e = embed(scoreBuf.copyOf()) ?: return null
        lastQueryEmbedding = e
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
    fun reset() { enrollmentMessage="Voice not learned (cleared/interrupted)"; lastQueryEmbedding = null; enrollmentEnabled = false; prints.clear(); centroid = null; enrollFill = 0; scoreFill = 0 }
}
