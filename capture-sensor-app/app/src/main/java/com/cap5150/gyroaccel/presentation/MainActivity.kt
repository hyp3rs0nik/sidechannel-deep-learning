// MainActivity.kt

package com.cap5150.gyroaccel.presentation

import android.content.Context
import android.hardware.*
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
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
import androidx.compose.ui.unit.dp
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.GZIPOutputStream

@Composable
fun GyroAccelTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        content = content
    )
}

class MainActivity : ComponentActivity(), SensorEventListener {
    private val SERVER_URL = "http://192.168.1.139:3000" // Updated server URL
    private var BATCH_INTERVAL_MS = 250L // Adjusted batching interval
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private var latestAccelValues = floatArrayOf(0f, 0f, 0f)
    private var latestGyroValues = floatArrayOf(0f, 0f, 0f)

    private var isTracking: MutableState<Boolean> = mutableStateOf(false)
    private val isBufferingData = AtomicBoolean(false)

    private val accumulatedSensorData = ConcurrentLinkedQueue<String>()
    private val handler = Handler(Looper.getMainLooper())

    // Time synchronization variables
    private var timeOffsetMs = 0L
    private var syncSensorTimeMs: Long = 0L
    private var syncServerTimeMs: Long = 0L
    private val syncIntervalMs = 5 * 60 * 1000L // Every 5 minutes

    // HandlerThread for sensor events
    private lateinit var sensorHandlerThread: HandlerThread
    private lateinit var sensorHandler: Handler

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Create a HandlerThread for sensor events
        sensorHandlerThread = HandlerThread("SensorThread")
        sensorHandlerThread.start()
        sensorHandler = Handler(sensorHandlerThread.looper)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            WearApp(isTracking, ::toggleTracking)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Quit the sensorHandlerThread
        sensorHandlerThread.quitSafely()
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
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST, sensorHandler)
        }
        gyroscope?.also { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST, sensorHandler)
        }

        synchronizeTimeWithServer()
        scheduleTimeSynchronization()
        handler.postDelayed(object : Runnable {
            override fun run() {
                if (isTracking.value) {
                    sendBatchedDataToServer()
                    handler.postDelayed(this, BATCH_INTERVAL_MS)
                }
            }
        }, BATCH_INTERVAL_MS)
    }

    private fun scheduleTimeSynchronization() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                if (isTracking.value) {
                    synchronizeTimeWithServer()
                    handler.postDelayed(this, syncIntervalMs)
                }
            }
        }, syncIntervalMs)
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
                        latestAccelValues = eventData.values.copyOf()
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        latestGyroValues = eventData.values.copyOf()
                    }
                }

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

        val dataToSendList = mutableListOf<String>()

        // Collect all available data
        while (true) {
            val data = accumulatedSensorData.poll() ?: break
            dataToSendList.add(data)
        }

        if (dataToSendList.isNotEmpty()) {
            isBufferingData.set(true)
            val dataToSend = dataToSendList.joinToString(separator = "\n")

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
                    dataToSendList.forEach { accumulatedSensorData.add(it) }
                    isBufferingData.set(false)
                }

                override fun onResponse(call: Call, response: Response) {
                    if (!response.isSuccessful) {
                        println("Unexpected response code: ${response.code}")
                        // Handle unsuccessful response, buffer data for retry
                        dataToSendList.forEach { accumulatedSensorData.add(it) }
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

    // Time synchronization methods
    private fun synchronizeTimeWithServer() {
        val client = OkHttpClient()
        val request = Request.Builder()
            .url("$SERVER_URL/server_time")
            .get()
            .build()

        val t0 = SystemClock.elapsedRealtimeNanos() / 1_000_000

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                println("Failed to synchronize time: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                val t3 = SystemClock.elapsedRealtimeNanos() / 1_000_000
                if (response.isSuccessful) {
                    val serverTimeJson = response.body?.string()
                    val serverTimeObject = JSONObject(serverTimeJson)
                    val t1 = serverTimeObject.getLong("timestamp")
                    val rtt = t3 - t0
                    val offset = t1 - t3 + rtt / 2
                    syncSensorTimeMs = t3
                    syncServerTimeMs = t1 + rtt / 2
                    timeOffsetMs = offset
                    println("Time offset calculated: $timeOffsetMs ms")
                }
            }
        })
    }

    private fun getAdjustedTimestamp(sensorTimestampNs: Long): Long {
        val sensorTimeMs = sensorTimestampNs / 1_000_000 // Convert to milliseconds
        val elapsedTimeSinceSyncMs = sensorTimeMs - syncSensorTimeMs
        val adjustedTimestamp = syncServerTimeMs + elapsedTimeSinceSyncMs
        return adjustedTimestamp
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
