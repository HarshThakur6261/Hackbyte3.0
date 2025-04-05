const authenticateToken = require("../Middleware/Authenticatetowken");
const ActiveChallenge = require("../Model/ActiveChallege");
const History = require("../Model/History");

const express = require("express")
const Historyrouter = express.Router();

Historyrouter.post("/create/:token", authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;

    // Find active challenge
    const activeChallenge = await ActiveChallenge.findOne({ 
      userId
    });

    if (!activeChallenge) {
      return res.status(404).json({ message: "No active challenge found" });
    }

    // Convert Mongoose document to plain object and remove _id
    const challengeData = activeChallenge.toObject();
    delete challengeData._id; // Let MongoDB generate a new _id for the history record

    // Create new history record directly from active challenge data
    const historyRecord = new History(challengeData);
    await historyRecord.save();

    res.status(201).json(historyRecord);
  } catch (error) {
    console.error("Error saving challenge history:", error);
    res.status(500).json({ 
      message: "Server error", 
      error: error.message 
    });
  }
});
Historyrouter.get("/get/:email", authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;

    const historyList = await History.find({ userId }).sort({ createdAt: -1 }); // Sort by recent first

    res.status(200).json(historyList);
  } catch (error) {
    res.status(500).json({ message: "Failed to fetch history", error });
  }
});


module.exports = Historyrouter;