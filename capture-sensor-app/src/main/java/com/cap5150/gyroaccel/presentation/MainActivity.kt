package com.cap5150.gyroaccel.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import com.cap5150.gyroaccel.presentation.theme.GyroAccelTheme
import java.util.Locale
import kotlin.math.abs
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Callback
import okhttp3.Call
import okhttp3.Response

class MainActivity : ComponentActivity(), SensorEventListener {
    private const val SERVER_URL = "http://192.168.1.139:3000"
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private lateinit var accelValues: MutableState<String>
    private lateinit var gyroValues: MutableState<String>


    private var prevAccelValues = floatArrayOf(0f, 0f, 0f)
    private var prevGyroValues = floatArrayOf(0f, 0f, 0f)
    private val threshold = 0.1f // threshold for significant change

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // initialize mutable state variables
        accelValues = mutableStateOf("Acc: 0, 0, 0")
        gyroValues = mutableStateOf("Gyro: 0, 0, 0")

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // set the app to keep the screen on
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            WearApp(accelValues, gyroValues)
        }
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.also { sensor: Sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
        }
        gyroscope?.also { sensor: Sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let { eventData: SensorEvent ->
            when (eventData.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    val x: Float = eventData.values[0]
                    val y: Float = eventData.values[1]
                    val z: Float = eventData.values[2]

                    if (isSignificantChange(x, y, z, prevAccelValues)) {
                        accelValues.value =
                            String.format(Locale.US, "Acc: %.3f, %.3f, %.3f", x, y, z)
                        prevAccelValues = floatArrayOf(x, y, z)

                        // Send updated accelerometer data
                        sendDataToServer(
                            accelX = x,
                            accelY = y,
                            accelZ = z,
                            gyroX = prevGyroValues[0],
                            gyroY = prevGyroValues[1],
                            gyroZ = prevGyroValues[2],
                            device = "right_wrist" // Change this based on the device location
                        )
                    }
                }

                Sensor.TYPE_GYROSCOPE -> {
                    val x: Float = eventData.values[0]
                    val y: Float = eventData.values[1]
                    val z: Float = eventData.values[2]

                    if (isSignificantChange(x, y, z, prevGyroValues)) {
                        gyroValues.value =
                            String.format(Locale.US, "Gyro: %.3f, %.3f, %.3f", x, y, z)
                        prevGyroValues = floatArrayOf(x, y, z)

                        // Send updated gyroscope data
                        sendDataToServer(
                            accelX = prevAccelValues[0],
                            accelY = prevAccelValues[1],
                            accelZ = prevAccelValues[2],
                            gyroX = x,
                            gyroY = y,
                            gyroZ = z,
                            device = "right_wrist" // Change this based on the device location
                        )
                    }
                }
            }
        }
    }


    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle accuracy changes if needed
    }

    private fun isSignificantChange(
        x: Float,
        y: Float,
        z: Float,
        previousValues: FloatArray
    ): Boolean {
        return (abs(x - previousValues[0]) > threshold ||
                abs(y - previousValues[1]) > threshold ||
                abs(z - previousValues[2]) > threshold)
    }

    private fun sendDataToServer(
        accelX: Float, accelY: Float, accelZ: Float,
        gyroX: Float, gyroY: Float, gyroZ: Float,
        device: String
    ) {
        val client = OkHttpClient()

        // Construct the JSON payload according to the server's expected schema
        val sensorData = """
        {
            "timestamp": ${System.currentTimeMillis()},
            "device": "$device",
            "accel_x": $accelX,
            "accel_y": $accelY,
            "accel_z": $accelZ,
            "gyro_x": $gyroX,
            "gyro_y": $gyroY,
            "gyro_z": $gyroZ
        }
    """.trimIndent()

        val mediaType = "application/json".toMediaType()
        val body: RequestBody = RequestBody.create(mediaType, sensorData)

        val request: Request = Request.Builder()
            .url(SERVER_URL + "/sensor-data")
            .post(body)
            .build()

        println("Sending data to server: $sensorData")

        // Send the request asynchronously
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                println("Failed to send data to server: ${e.message}")
            }

            @Throws(IOException::class)
            override fun onResponse(call: Call, response: Response) {
                println("Response received: ${response.body?.string()}")
            }
        })
    }
}

@Composable
fun WearApp(accelValues: MutableState<String>, gyroValues: MutableState<String>) {
    GyroAccelTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colors.background),
            contentAlignment = Alignment.Center
        ) {
            Column {
                Text(
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colors.primary,
                    text = accelValues.value
                )
                Text(
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colors.primary,
                    text = gyroValues.value
                )
            }
        }
    }
}
