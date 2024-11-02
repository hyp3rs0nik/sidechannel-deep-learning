package com.cap5150.gyroaccel.presentation

import android.app.AlertDialog
import android.content.Context
import android.hardware.*
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
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.*
import java.util.zip.GZIPOutputStream

class MainActivity : ComponentActivity(), SensorEventListener {
    private val SERVER_URL = "http://192.168.1.139:3000"
    private var BATCH_INTERVAL_MS = 1000L
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var rotationVectorSensor: Sensor? = null

    private var latestAccelValues = floatArrayOf(0f, 0f, 0f)
    private var latestGyroValues = floatArrayOf(0f, 0f, 0f)
    private var latestRotationVector = floatArrayOf(0f, 0f, 0f, 0f)

    private lateinit var accelValues: MutableState<String>
    private lateinit var gyroValues: MutableState<String>
    private var isTracking: MutableState<Boolean> = mutableStateOf(false)

    private val accumulatedSensorData = mutableListOf<String>()
    private val handler = Handler(Looper.getMainLooper())

    // Variables for calibration
    private var biasX = 0f
    private var biasY = 0f
    private var biasZ = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        accelValues = mutableStateOf("Accel: 0, 0, 0")
        gyroValues = mutableStateOf("Gyro: 0, 0, 0")

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            WearApp(accelValues, gyroValues, isTracking, ::toggleTracking)
        }

        // Start calibration process
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
        accelerometer?.also { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.also { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        }
        rotationVectorSensor?.also { sensor ->
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
                when (eventData.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        val (x, y, z) = eventData.values

                        // Apply calibration
                        val correctedValues = applyBiasRemoval(x, y, z)
                        val gravityCompensatedZ = applyGravityCompensation(correctedValues[2])

                        accelValues.value = String.format(
                            Locale.US,
                            "Accel: %.3f, %.3f, %.3f",
                            correctedValues[0],
                            correctedValues[1],
                            gravityCompensatedZ
                        )
                        latestAccelValues = floatArrayOf(
                            correctedValues[0],
                            correctedValues[1],
                            gravityCompensatedZ
                        )
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        val (x, y, z) = eventData.values
                        gyroValues.value = String.format(
                            Locale.US,
                            "Gyro: %.3f, %.3f, %.3f",
                            x,
                            y,
                            z
                        )
                        latestGyroValues = floatArrayOf(x, y, z)
                    }
                    Sensor.TYPE_ROTATION_VECTOR -> {
                        val rotationVector = FloatArray(4)
                        SensorManager.getQuaternionFromVector(rotationVector, eventData.values)
                        latestRotationVector = rotationVector
                    }
                }
            }
            accumulateSensorData()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun accumulateSensorData() {
        val timestamp = System.currentTimeMillis()
        val dataMap = mapOf(
            "timestamp" to timestamp,
            "accel_x" to latestAccelValues[0],
            "accel_y" to latestAccelValues[1],
            "accel_z" to latestAccelValues[2],
            "gyro_x" to latestGyroValues[0],
            "gyro_y" to latestGyroValues[1],
            "gyro_z" to latestGyroValues[2],
            "rotation_x" to latestRotationVector[0],
            "rotation_y" to latestRotationVector[1],
            "rotation_z" to latestRotationVector[2],
            "rotation_w" to latestRotationVector[3]
        )
        val dataJson = JSONObject(dataMap).toString()
        accumulatedSensorData.add(dataJson)
    }

    private fun sendBatchedDataToServer() {
        if (accumulatedSensorData.isNotEmpty()) {
            val dataToSend = accumulatedSensorData.joinToString(separator = "\n")
            accumulatedSensorData.clear()

            // Compress the data
            val compressedData = compressData(dataToSend)
            println("Compressed data size: ${compressedData.size} bytes") // Log compressed data size

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
                }

                override fun onResponse(call: Call, response: Response) {
                    if (!response.isSuccessful) throw IOException("Unexpected code $response")
                    println("Batched data response received: ${response.body?.string()}")
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
        val calibrationSamples = 200
        val calibrationInterval = 10L // milliseconds

        sensorManager.registerListener(object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                event?.let {
                    if (it.sensor.type == Sensor.TYPE_ACCELEROMETER) {
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

    private fun applyGravityCompensation(z: Float): Float {
        val gravity = 9.81f // Gravity constant
        return z - gravity
    }
}

@Composable
fun WearApp(
    accelValues: MutableState<String>,
    gyroValues: MutableState<String>,
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
