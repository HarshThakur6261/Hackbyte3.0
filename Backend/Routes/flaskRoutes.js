const express = require("express");
const router = express.Router();
const axios = require("axios");

const FLASK_SERVER_URL = "http://localhost:3001"; // Update if needed

// Start Squat Tracking
router.get("/start_squat", async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_SERVER_URL}/start_squat`);
    console.log(response.data);
    res.json(response.data);
  } catch (error) {
    console.error("Error starting squat tracking:", error.message);
    res.status(500).json({ error: "Failed to start squat tracking" });
  }
});

// Stop Squat Tracking
router.get("/stop_squat", async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_SERVER_URL}/stop_squat`);
    console.log("stop", response.data);
    res.json(response.data);
  } catch (error) {
    console.error("Error stopping squat tracking:", error.message);
    res.status(500).json({ error: "Failed to stop squat tracking" });
  }
});

// Get Squat Count
router.get("/get_squat_count", async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_SERVER_URL}/get_squat_count`);
    console.log("count", response.data);
    res.json(response.data);
  } catch (error) {
    console.error("Error fetching squat count:", error.message);
    res.status(500).json({ error: "Failed to fetch squat count" });
  }
});

// Start Pushup Tracking
router.get("/start_pushup", async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_SERVER_URL}/start_pushup`);
    console.log(response.data);
    res.json(response.data);
  } catch (error) {
    console.error("Error starting pushup tracking:", error.message);
    res.status(500).json({ error: "Failed to start pushup tracking" });
  }
});

// Stop Pushup Tracking
router.get("/stop_pushup", async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_SERVER_URL}/stop_pushup`);
    console.log("stop", response.data);
    res.json(response.data);
  } catch (error) {
    console.error("Error stopping pushup tracking:", error.message);
    res.status(500).json({ error: "Failed to stop pushup tracking" });
  }
});

// Get Pushup Count
router.get("/get_pushup_count", async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_SERVER_URL}/get_pushup_count`);
    console.log("count", response.data);
    res.json(response.data);
  } catch (error) {
    console.error("Error fetching pushup count:", error.message);
    res.status(500).json({ error: "Failed to fetch pushup count" });
  }
});
router.post('/recommendations', async (req, res) => {
  try {
    console.log("hello")
    // Verify the data exists and has required fields
    if (!req.body || !req.body.height) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const response = await axios.post('http://127.0.0.1:5000/recommend', req.body, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    console.log(response.data)
    
    res.json(response.data);
  } catch (error) {
    console.error('Flask API error:', error);
    res.status(error.response?.status || 500).json({ 
      error: error.response?.data?.error || 'Failed to get recommendations' 
    });
  }
});

module.exports = router;




