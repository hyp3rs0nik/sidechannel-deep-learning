// MainActivity.kt

package com.cap5150.gyroaccel.presentation

import android.app.AlertDialog
import android.content.Context
import android.hardware.*
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import com.cap5150.gyroaccel.presentation.theme.GyroAccelTheme
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.GZIPOutputStream

class MainActivity : ComponentActivity(), SensorEventListener {
    private val SERVER_URL = "http://192.168.1.139:3000" // Updated server URL
    private var BATCH_INTERVAL_MS = 250L // Adjusted batching interval
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private var latestAccelValues = floatArrayOf(0f, 0f, 0f)
    private var previousAccelValues = floatArrayOf(0f, 0f, 0f)
    private var latestGyroValues = floatArrayOf(0f, 0f, 0f)
    private var previousGyroValues = floatArrayOf(0f, 0f, 0f)

    private var isTracking: MutableState<Boolean> = mutableStateOf(false)
    private val isBufferingData = AtomicBoolean(false)

    private val accumulatedSensorData = mutableListOf<String>()
    private val handler = Handler(Looper.getMainLooper())

    // Variables for calibration
    private var biasX = 0f
    private var biasY = 0f
    private var biasZ = 0f

    // Time synchronization variables
    private var timeOffsetMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            WearApp(isTracking, ::toggleTracking)
        }

        startCalibration()
    }

    private fun toggleTracking() {
        if (isTracking.value) {
            stopTracking()
        } else {
            startTracking()
        }
    }

    private fun startTracking() {
        isTracking.value = true
        val desiredSamplingRateUs = (1_000_000 / 200) // 200 Hz

        accelerometer?.also { sensor ->
            sensorManager.registerListener(this, sensor, desiredSamplingRateUs)
        }
        gyroscope?.also { sensor ->
            sensorManager.registerListener(this, sensor, desiredSamplingRateUs)
        }

        synchronizeTimeWithServer()
        handler.postDelayed(object : Runnable {
            override fun run() {
                if (isTracking.value) {
                    sendBatchedDataToServer()
                    handler.postDelayed(this, BATCH_INTERVAL_MS)
                }
            }
        }, BATCH_INTERVAL_MS)
    }

    private fun stopTracking() {
        isTracking.value = false
        sensorManager.unregisterListener(this)
        handler.removeCallbacksAndMessages(null)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (isTracking.value) {
            event?.let { eventData ->
                val adjustedTimestamp = getAdjustedTimestamp(eventData.timestamp)

                when (eventData.sensor.type) {
                    Sensor.TYPE_LINEAR_ACCELERATION -> {
                        val (x, y, z) = eventData.values

                        val correctedValues = applyBiasRemoval(x, y, z)
                        val filteredValues = applyLowPassFilter(correctedValues, previousAccelValues, alpha = 0.8f)
                        previousAccelValues = filteredValues.copyOf()

                        latestAccelValues = filteredValues
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        val (x, y, z) = eventData.values

                        val filteredValues = applyLowPassFilter(eventData.values, previousGyroValues, alpha = 0.8f)
                        previousGyroValues = filteredValues.copyOf()

                        latestGyroValues = filteredValues
                    }
                }

                // Accumulate data only when both accelerometer and gyroscope have updated values
                accumulateSensorData(adjustedTimestamp)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun accumulateSensorData(timestamp: Long) {
        val dataMap = mapOf(
            "timestamp" to timestamp,
            "accel_x" to latestAccelValues[0],
            "accel_y" to latestAccelValues[1],
            "accel_z" to latestAccelValues[2],
            "gyro_x" to latestGyroValues[0],
            "gyro_y" to latestGyroValues[1],
            "gyro_z" to latestGyroValues[2],
        )
        val dataJson = JSONObject(dataMap).toString()
        accumulatedSensorData.add(dataJson)
    }

    private fun sendBatchedDataToServer() {
        if (isBufferingData.get()) return

        if (accumulatedSensorData.isNotEmpty()) {
            isBufferingData.set(true)
            val dataToSend = accumulatedSensorData.joinToString(separator = "\n")
            accumulatedSensorData.clear()

            // Compress the data
            val compressedData = compressData(dataToSend)

            val client = OkHttpClient()
            val body: RequestBody = compressedData.toRequestBody("application/octet-stream".toMediaType())
            val request = Request.Builder()
                .url("$SERVER_URL/sensor_data")
                .post(body)
                .addHeader("Content-Encoding", "gzip")
                .build()

            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    println("Failed to send batched data: ${e.message}")
                    // Handle network failure, buffer data for retry
                    accumulatedSensorData.add(dataToSend)
                    isBufferingData.set(false)
                }

                override fun onResponse(call: Call, response: Response) {
                    if (!response.isSuccessful) {
                        println("Unexpected response code: ${response.code}")
                        // Handle unsuccessful response, buffer data for retry
                        accumulatedSensorData.add(dataToSend)
                    }
                    isBufferingData.set(false)
                }
            })
        }
    }

    private fun compressData(data: String): ByteArray {
        val outputStream = ByteArrayOutputStream()
        GZIPOutputStream(outputStream).use { it.write(data.toByteArray(Charsets.UTF_8)) }
        return outputStream.toByteArray()
    }

    // Calibration methods
    private fun startCalibration() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Calibration")
        builder.setMessage("Place the device on a stable surface and tap OK to start calibration.")
        builder.setCancelable(false)
        builder.setPositiveButton("OK") { dialog, _ ->
            dialog.dismiss()
            performCalibration()
        }
        val dialog = builder.create()
        dialog.show()
    }

    private fun performCalibration() {
        val calibrationData = mutableListOf<FloatArray>()
        val calibrationSamples = 500 // Increased samples for better accuracy

        sensorManager.registerListener(object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                event?.let {
                    if (it.sensor.type == Sensor.TYPE_LINEAR_ACCELERATION) {
                        calibrationData.add(it.values.clone())
                        if (calibrationData.size >= calibrationSamples) {
                            sensorManager.unregisterListener(this)
                            computeCalibrationBias(calibrationData)
                        }
                    }
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)
    }

    private fun computeCalibrationBias(calibrationData: List<FloatArray>) {
        val sumX = calibrationData.sumOf { it[0].toDouble() }
        val sumY = calibrationData.sumOf { it[1].toDouble() }
        val sumZ = calibrationData.sumOf { it[2].toDouble() }

        biasX = (sumX / calibrationData.size).toFloat()
        biasY = (sumY / calibrationData.size).toFloat()
        biasZ = (sumZ / calibrationData.size).toFloat()
    }

    private fun applyBiasRemoval(x: Float, y: Float, z: Float): FloatArray {
        return floatArrayOf(x - biasX, y - biasY, z - biasZ)
    }

    private fun applyLowPassFilter(
        currentValues: FloatArray,
        previousValues: FloatArray,
        alpha: Float
    ): FloatArray {
        val filteredValues = FloatArray(3)
        for (i in 0..2) {
            filteredValues[i] = alpha * currentValues[i] + (1 - alpha) * previousValues[i]
        }
        return filteredValues
    }

    // Time synchronization methods
    private fun synchronizeTimeWithServer() {
        val client = OkHttpClient()
        val request = Request.Builder()
            .url("$SERVER_URL/server_time")
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                println("Failed to synchronize time: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val serverTimeJson = response.body?.string()
                    val serverTimeObject = JSONObject(serverTimeJson)
                    val serverTimeMs = serverTimeObject.getLong("timestamp")
                    val clientTimeMs = System.currentTimeMillis()
                    timeOffsetMs = serverTimeMs - clientTimeMs
                    println("Time offset calculated: $timeOffsetMs ms")
                }
            }
        })
    }

    private fun getAdjustedTimestamp(sensorTimestampNs: Long): Long {
        val bootTimeMs = System.currentTimeMillis() - SystemClock.elapsedRealtime()
        val sensorTimeMs = bootTimeMs + (sensorTimestampNs / 1_000_000)
        return sensorTimeMs + timeOffsetMs
    }
}

@Composable
fun WearApp(
    isTracking: MutableState<Boolean>,
    onToggleTracking: () -> Unit
) {
    GyroAccelTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colors.background),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Button(
                    onClick = { onToggleTracking() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                ) {
                    Text(text = if (isTracking.value) "Stop Tracking" else "Start Tracking")
                }
            }
        }
    }
}
