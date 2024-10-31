package com.cap5150.gyroaccel.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
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
import java.io.IOException
import java.util.Locale

class MainActivity : ComponentActivity(), SensorEventListener {
    private val SERVER_URL = "http://192.168.1.139:3000"
    private val BATCH_INTERVAL_MS = 1000L
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null

    private var latestAccelValues = floatArrayOf(0f, 0f, 0f)
    private lateinit var accelValues: MutableState<String>
    private var isTracking: MutableState<Boolean> = mutableStateOf(false)

    private val accumulatedSensorData = mutableListOf<String>()
    private val handler = Handler(Looper.getMainLooper())

    // Variables for calibration
    private var biasX = 0f
    private var biasY = 0f
    private var biasZ = 0f
    private val alpha = 0.8f  // Low-pass filter parameter
    private var smoothedValues = floatArrayOf(0f, 0f, 0f)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        accelValues = mutableStateOf("Accel: 0, 0, 0")

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            WearApp(accelValues, isTracking, ::toggleTracking)
        }

        // Perform sensor calibration at startup
        calibrateSensor()
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
        accelerometer?.also { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        }
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
                if (eventData.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                    val (x, y, z) = eventData.values

                    // Apply calibration
                    val correctedValues = applyBiasRemoval(x, y, z)
                    val gravityCompensatedZ = applyGravityCompensation(correctedValues[2])
                    val smoothedAccelValues = applyLowPassFilter(floatArrayOf(correctedValues[0], correctedValues[1], gravityCompensatedZ))

                    accelValues.value = String.format(Locale.US, "Accel: %.3f, %.3f, %.3f", smoothedAccelValues[0], smoothedAccelValues[1], smoothedAccelValues[2])
                    latestAccelValues = smoothedAccelValues

                    accumulateSensorData(smoothedAccelValues[0], smoothedAccelValues[1], smoothedAccelValues[2])
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun accumulateSensorData(x: Float, y: Float, z: Float) {
        val timestamp = System.currentTimeMillis()
        val data = "$timestamp, $x, $y, $z"
        accumulatedSensorData.add(data)
    }

    private fun sendBatchedDataToServer() {
        if (accumulatedSensorData.isNotEmpty()) {
            val dataToSend = accumulatedSensorData.joinToString(separator = "\n") {
                it.replace(", ", ",")
            }

            accumulatedSensorData.clear()

            val client = OkHttpClient()
            val body: RequestBody = dataToSend.toRequestBody("text/csv".toMediaType())
            val request = Request.Builder()
                .url("$SERVER_URL/sensor_data")
                .post(body)
                .build()

            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    println("Failed to send batched data: ${e.message}")
                }

                override fun onResponse(call: Call, response: Response) {
                    if (!response.isSuccessful) throw IOException("Unexpected code $response")
                    println("Batched data response received: ${response.body?.string()}")
                }
            })
        }
    }

    // Calibration methods
    private fun calibrateSensor() {
        val stationaryData = mutableListOf<FloatArray>()
        repeat(100) {  // Collect data 100 times
            stationaryData.add(latestAccelValues)
            Thread.sleep(10)  // Pause briefly between readings
        }

        val sumX = stationaryData.sumOf { it[0].toDouble() }
        val sumY = stationaryData.sumOf { it[1].toDouble() }
        val sumZ = stationaryData.sumOf { it[2].toDouble() }

        biasX = (sumX / stationaryData.size).toFloat()
        biasY = (sumY / stationaryData.size).toFloat()
        biasZ = (sumZ / stationaryData.size).toFloat()
    }

    private fun applyBiasRemoval(x: Float, y: Float, z: Float): FloatArray {
        return floatArrayOf(x - biasX, y - biasY, z - biasZ)
    }

    private fun applyGravityCompensation(z: Float): Float {
        val gravity = 9.81f  // Gravity constant
        return z - gravity
    }

    private fun applyLowPassFilter(newValues: FloatArray): FloatArray {
        smoothedValues[0] = alpha * smoothedValues[0] + (1 - alpha) * newValues[0]
        smoothedValues[1] = alpha * smoothedValues[1] + (1 - alpha) * newValues[1]
        smoothedValues[2] = alpha * smoothedValues[2] + (1 - alpha) * newValues[2]
        return smoothedValues
    }
}

@Composable
fun WearApp(
    accelValues: MutableState<String>,
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
