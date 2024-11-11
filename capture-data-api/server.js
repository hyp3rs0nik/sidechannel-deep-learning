const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const morgan = require("morgan");
const zlib = require("zlib");
const bodyParser = require("body-parser");
const { Schema, model, connection } = mongoose;

const app = express();
app.use(cors());
app.use(morgan("tiny"));
app.use(bodyParser.json({ limit: "10mb" }));

const connectWithRetry = async () => {
  try {
    await mongoose.connect("mongodb://localhost:27017/v1", {
      maxPoolSize: 10,
    });
    console.log("Connected to MongoDB.");
  } catch (err) {
    console.error("DB connection error:", err);
    setTimeout(connectWithRetry, 5000);
  }
};
connectWithRetry();

const sensorSchema = new Schema({
  timestamp: { type: Number, required: true },
  accel_x: { type: Number, required: true },
  accel_y: { type: Number, required: true },
  accel_z: { type: Number, required: true },
  gyro_x: { type: Number, required: true },
  gyro_y: { type: Number, required: true },
  gyro_z: { type: Number, required: true },
});

sensorSchema.index({ timestamp: 1 });

const Sensor = model("Sensor", sensorSchema);

const keystrokeSchema = new Schema({
  sequenceIndex: { type: Number, required: true },
  sequence: { type: [String], required: true },
  key: { type: String, required: true },
  timestamp: { type: Number, required: true },
  eventType: { type: String, required: true },
  cursorPosition: { type: Number, required: true },
  inputValue: { type: String, required: true },
});

keystrokeSchema.index({ timestamp: 1 });

const Keystroke = model("Keystroke", keystrokeSchema);

const bulkInsertSensorData = async (docs) => {
  if (!docs.length) return;
  try {
    await Sensor.insertMany(docs, { ordered: false });
    console.log(`${docs.length} sensor records inserted.`);
  } catch (err) {
    console.error('Error inserting sensor data:', err);
  }
};

const bulkInsertKeystrokeData = async (docs) => {
  if (!docs.length) return;
  try {
    await Keystroke.insertMany(docs, { ordered: false });
    console.log(`${docs.length} keystroke records inserted.`);
  } catch (err) {
    console.error("Error inserting keystroke data:", err);
  }
};

app.get("/server_time", (req, res) => {
  const serverTime = Date.now();
  res.json({ timestamp: serverTime });
});

app.post("/sensor_data", (req, res) => {
  const chunks = [];
  req.on("data", (chunk) => {
    chunks.push(chunk);
  });
  req.on("end", () => {
    const buffer = Buffer.concat(chunks);
    zlib.gunzip(buffer, (err, decoded) => {
      if (err) {
        console.error("Error decompressing data:", err);
        res.status(400).send("Error decompressing data");
        return;
      }
      const dataStr = decoded.toString();

      try {
        const dataArray = dataStr
          .split("\n")
          .map((line) => JSON.parse(line.trim()));

        const sensorDocs = dataArray.map((data) => ({
          timestamp: data.timestamp,
          accel_x: data.accel_x,
          accel_y: data.accel_y,
          accel_z: data.accel_z,
          gyro_x: data.gyro_x,
          gyro_y: data.gyro_y,
          gyro_z: data.gyro_z,
        }));

        bulkInsertSensorData(sensorDocs);

        res.status(200).send("OK");
      } catch (parseErr) {
        console.error("Error parsing JSON data:", parseErr);
        res.status(400).send("Invalid JSON data format");
      }
    });
  });
});

app.post("/keystroke_data", async (req, res) => {
  try {
    const keystrokeData = req.body;

    if (!Array.isArray(keystrokeData)) {
      throw new Error("Keystroke data should be an array");
    }

    const keystrokeDocs = keystrokeData.map((data) => ({
      sequenceIndex: data.sequenceIndex,
      sequence: data.sequence,
      key: data.key,
      timestamp: data.timestamp,
      eventType: data.eventType,
      cursorPosition: data.cursorPosition,
      inputValue: data.inputValue,
    }));

    await bulkInsertKeystrokeData(keystrokeDocs);

    res
      .status(200)
      .json({ message: "Keystroke data received and stored successfully" });
  } catch (err) {
    console.error("Error handling keystroke data:", err);
    res.status(400).send("Invalid keystroke data format");
  }
});

process.on("SIGINT", async () => {
  console.log("Shutting down gracefully...");
  await connection.close();
  process.exit(0);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}.`));
