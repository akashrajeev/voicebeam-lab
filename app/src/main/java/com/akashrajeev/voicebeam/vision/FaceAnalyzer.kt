package com.akashrajeev.voicebeam.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.akashrajeev.voicebeam.core.Box
import com.akashrajeev.voicebeam.core.FaceObservation
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.hypot

/**
 * CameraX analyzer that runs MediaPipe Face Landmarker on every frame and
 * reports faces as boxes plus a mouth-openness value (inner-lip gap divided
 * by face height), all in upright, normalised image coordinates.
 */
class FaceAnalyzer(
    context: Context,
    private val onFaces: (timeMs: Long, faces: List<FaceObservation>, width: Int, height: Int) -> Unit,
) : ImageAnalysis.Analyzer {

    private var landmarker: FaceLandmarker? = null
    @Volatile private var lastW = 0
    @Volatile private var lastH = 0
    @Volatile private var busy = false

    init {
        landmarker = create(context, Delegate.GPU) ?: create(context, Delegate.CPU)
    }

    private fun create(context: Context, delegate: Delegate): FaceLandmarker? = try {
        val base = BaseOptions.builder().setModelAssetPath("models/face_landmarker.task").setDelegate(delegate).build()
        val opts = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(4)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { r: FaceLandmarkerResult, _ -> handle(r) }
            .setErrorListener { e -> Log.w("VoiceBeamVision", "landmarker error", e); busy = false }
            .build()
        FaceLandmarker.createFromOptions(context, opts)
    } catch (t: Throwable) {
        Log.w("VoiceBeamVision", "landmarker init failed on $delegate", t); null
    }

    override fun analyze(image: ImageProxy) {
        val lm = landmarker
        if (lm == null || busy) { image.close(); return }
        try {
            val bmp = image.toBitmap()
            val rot = image.imageInfo.rotationDegrees
            val upright = if (rot != 0) {
                val m = Matrix().apply { postRotate(rot.toFloat()) }
                Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            } else bmp
            lastW = upright.width; lastH = upright.height
            busy = true
            lm.detectAsync(BitmapImageBuilder(upright).build(), SystemClock.uptimeMillis())
        } catch (t: Throwable) {
            busy = false
            Log.w("VoiceBeamVision", "analyze failed", t)
        } finally {
            image.close()
        }
    }

    private fun handle(r: FaceLandmarkerResult) {
        busy = false
        val faces = r.faceLandmarks().map { pts ->
            var minX = 1f; var minY = 1f; var maxX = 0f; var maxY = 0f
            for (p in pts) {
                minX = minOf(minX, p.x()); minY = minOf(minY, p.y())
                maxX = maxOf(maxX, p.x()); maxY = maxOf(maxY, p.y())
            }
            val faceH = hypot(pts[10].x() - pts[152].x(), pts[10].y() - pts[152].y()).coerceAtLeast(1e-4f)
            val lipGap = hypot(pts[13].x() - pts[14].x(), pts[13].y() - pts[14].y())
            FaceObservation(Box(minX, minY, maxX, maxY), lipGap / faceH)
        }
        onFaces(SystemClock.uptimeMillis(), faces, lastW, lastH)
    }

    val available: Boolean get() = landmarker != null

    fun close() { landmarker?.close(); landmarker = null }
}
