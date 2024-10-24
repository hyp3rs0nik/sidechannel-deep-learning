const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const morgan = require('morgan');
const { Schema, model, connection } = mongoose;

const app = express();
app.use(express.json({ limit: '5mb' }));
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

const sensorSchema = new Schema(
  {
    timestamp: { type: Number, required: true },
    x: { type: Number, required: true },
    y: { type: Number, required: true },
    z: { type: Number, required: true },
  },
  { timestamps: false }
);

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

app.post('/sensor_data', express.raw({ type: 'text/csv' }), async (req, res) => {
  try {
    const data = req.body.toString().trim().split('\n');
    const sensorDocs = data.map((row) => {
      const [timestamp, x, y, z] = row.split(',');
      return {
        timestamp: Number(timestamp),
        x: parseFloat(x),
        y: parseFloat(y),
        z: parseFloat(z),
      };
    });

    await bulkInsert(sensorDocs);
    res.status(200).send('Sensor data received.');
  } catch (err) {
    console.error('Error processing sensor data:', err);
    res.status(500).send('Server error.');
  }
});

// Graceful shutdown handling
process.on('SIGINT', async () => {
  console.log('Shutting down gracefully...');
  await connection.close();
  process.exit(0);
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}.`));
