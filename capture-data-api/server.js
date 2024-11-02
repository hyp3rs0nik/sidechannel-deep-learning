const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const morgan = require('morgan');
const zlib = require('zlib');
const { Schema, model, connection } = mongoose;

const app = express();
app.use(cors());
app.use(morgan('tiny'));

const connectWithRetry = async () => {
  try {
    await mongoose.connect('mongodb://localhost:27017/side-channel', {
      maxPoolSize: 10,
    });
    console.log('Connected to MongoDB.');
  } catch (err) {
    console.error('DB connection error:', err);
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
  rotation_x: { type: Number, required: true },
  rotation_y: { type: Number, required: true },
  rotation_z: { type: Number, required: true },
  rotation_w: { type: Number, required: true },
});

const Sensor = model('Sensor', sensorSchema);

const bulkInsert = async (docs) => {
  if (!docs.length) return;
  try {
    await Sensor.insertMany(docs, { ordered: false });
    console.log(`${docs.length} sensor records inserted.`);
  } catch (err) {
    console.error('Error inserting sensor data:', err);
  }
};

app.post('/sensor_data', (req, res) => {
  const chunks = [];
  req.on('data', (chunk) => {
      chunks.push(chunk);
  });
  req.on('end', () => {
      const buffer = Buffer.concat(chunks);
      zlib.gunzip(buffer, (err, decoded) => {
        if (err) {
            console.error('Error decompressing data:', err);
            res.status(400).send('Error decompressing data');
            return;
        }
        const dataStr = decoded.toString();
        
        try {
            const dataArray = dataStr.split('\n').map((line) => JSON.parse(line.trim()));

            const sensorDocs = dataArray.map((data) => ({
                timestamp: data.timestamp,
                accel_x: data.accel_x,
                accel_y: data.accel_y,
                accel_z: data.accel_z,
                gyro_x: data.gyro_x,
                gyro_y: data.gyro_y,
                gyro_z: data.gyro_z,
                rotation_x: data.rotation_x,
                rotation_y: data.rotation_y,
                rotation_z: data.rotation_z,
                rotation_w: data.rotation_w,
            }));

            bulkInsert(sensorDocs);

            res.status(200).send('OK');
        } catch (parseErr) {
            console.error('Error parsing JSON data:', parseErr);
            res.status(400).send('Invalid JSON data format');
        }
    });
  });
});
          
process.on('SIGINT', async () => {
  console.log('Shutting down gracefully...');
  await connection.close();
  process.exit(0);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}.`));
