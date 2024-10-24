const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const csvParser = require('csv-parser');

const app = express();

app.use(express.text({ limit: '10mb', type: 'text/csv' }));
app.use(cors());

mongoose.connect('mongodb://localhost:27017/side-channel', {
  maxPoolSize: 10,
}).then(() => {
  console.log('Connected to database');

  const sensorSchema = new mongoose.Schema({
    timestamp: Number,
    x: Number,
    y: Number,
    z: Number,
  });

  sensorSchema.index({ timestamp: 1 });

  const Sensor = mongoose.model('Sensor', sensorSchema);

  app.post('/sensor_data', async (req, res) => {
    try {
      const csvData = req.body;

      if (!csvData || typeof csvData !== 'string') {
        return res.status(400).send('Expected CSV data in the request body.');
      }

      // Parse CSV data
      const sensorDocs = [];
      const parser = csvParser({
        headers: ['timestamp', 'x', 'y', 'z'],
        skipLines: 0,
      });

      parser.on('data', (data) => {
        const parsedData = {
          timestamp: Number(data.timestamp),
          x: Number(data.x),
          y: Number(data.y),
          z: Number(data.z),
        };
        if (!Object.values(parsedData).some(isNaN)) {
          sensorDocs.push(parsedData);
        }
      });

      parser.on('end', async () => {
        if (sensorDocs.length === 0) {
          return res.status(400).send('No valid sensor data to insert.');
        }

        await Sensor.insertMany(sensorDocs, { ordered: false });

        res.status(200).send('Sensor data saved');
        console.log(`Sensor data saved: ${sensorDocs.length} entries`);
      });

      parser.on('error', (err) => {
        console.error('Error parsing CSV data:', err);
        res.status(500).send('Error parsing CSV data');
      });

      const stream = require('stream');
      const Readable = stream.Readable;
      const s = new Readable();
      s.push(csvData);
      s.push(null);
      s.pipe(parser);
    } catch (err) {
      console.error('Error saving sensor data:', err);
      res.status(500).send('Error saving sensor data');
    }
  });

  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`Data capture server running on port ${PORT}`);
  });
}).catch((err) => {
  console.error('DB connection error, aborting:', err);
  process.exit(1);
});
