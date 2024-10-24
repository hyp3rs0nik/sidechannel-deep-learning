package com.cap5150.gyroaccel.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.IOException
import java.util.Locale

class MainActivity : ComponentActivity(), SensorEventListener {
    private val SERVER_URL = "http://192.168.1.139:3000"
    private val BATCH_INTERVAL_MS = 5000L
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null

    private var latestAccelValues = FloatArray(3) { 0f }
    private var latestGyroValues = FloatArray(3) { 0f }
    private var latestAccelTimestamp: Long = 0L
    private var latestGyroTimestamp: Long = 0L

    private lateinit var accelValues: MutableState<String>
    private var isTracking: MutableState<Boolean> = mutableStateOf(false)

    private val accumulatedSensorData = mutableListOf<SensorData>()
    private val mainScope = CoroutineScope(Dispatchers.Main + Job())

    private var timestampOffset: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        accelValues = mutableStateOf("Accel: 0.000, 0.000, 0.000")
        gyroValues = mutableStateOf("Gyro: 0.000, 0.000, 0.000")

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            WearApp(accelValues, isTracking, ::toggleTracking)
        }
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
        calculateTimestampOffset()
        accelerometer?.also { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.also { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        }
        mainScope.launch {
            while (isTracking.value) {
                delay(BATCH_INTERVAL_MS)
                sendBatchedDataToServer()
            }
        }
    }

    private fun stopTracking() {
        isTracking.value = false
        sensorManager.unregisterListener(this)
        mainScope.cancel()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (isTracking.value) {
            event?.let { eventData ->
                val adjustedTimestamp =
                    eventData.timestamp / 1_000_000 + timestampOffset // Convert to ms
                when (eventData.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        latestAccelValues = eventData.values.copyOf()
                        latestAccelTimestamp = adjustedTimestamp
                        accelValues.value = String.format(
                            Locale.US, "Accel: %.3f, %.3f, %.3f",
                            latestAccelValues[0], latestAccelValues[1], latestAccelValues[2]
                        )
                    }

                    Sensor.TYPE_GYROSCOPE -> {
                        latestGyroValues = eventData.values.copyOf()
                        latestGyroTimestamp = adjustedTimestamp
                        gyroValues.value = String.format(
                            Locale.US, "Gyro: %.3f, %.3f, %.3f",
                            latestGyroValues[0], latestGyroValues[1], latestGyroValues[2]
                        )
                    }
                }
                accumulateSensorData(
                    adjustedTimestamp,
                    latestGyroValues[0],
                    latestGyroValues[1],
                    latestGyroValues[2],
                    latestAccelValues[0],
                    latestAccelValues[1],
                    latestAccelValues[2]
                )
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {

    }

    private fun calculateTimestampOffset() {
        timestampOffset =
            System.currentTimeMillis() - SystemClock.elapsedRealtimeNanos() / 1_000_000
    }

    private fun accumulateSensorData(
        timestamp: Long,
        gyroX: Float,
        gyroY: Float,
        gyroZ: Float,
        accelX: Float,
        accelY: Float,
        accelZ: Float
    ) {
        val data = SensorData(
            timestamp = timestamp,
            gyroX = gyroX,
            gyroY = gyroY,
            gyroZ = gyroZ,
            accelX = accelX,
            accelY = accelY,
            accelZ = accelZ
        )
        accumulatedSensorData.add(data)
    }

    private fun sendBatchedDataToServer() {
        if (accumulatedSensorData.isNotEmpty()) {
            val dataToSend = buildString {
                accumulatedSensorData.forEach { data ->
                    append("${data.timestamp},${data.gyroX},${data.gyroY},${data.gyroZ},${data.accelX},${data.accelY},${data.accelZ}\n")
                }
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
                    Log.e("cap5150", "Failed to send data to server", e)
                }

                override fun onResponse(call: Call, response: Response) {
                    response.close()
                }
            })
        }
    }
}

@Serializable
data class SensorData(
    val timestamp: Long,
    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float,
    val accelX: Float,
    val accelY: Float,
    val accelZ: Float
)

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
                Text(
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colors.primary,
                    text = accelValues.value
                )
                Spacer(modifier = Modifier.height(20.dp))
                Button(
                    onClick = onToggleTracking,
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
