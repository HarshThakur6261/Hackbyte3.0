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


// Historyrouter.get('/user/:token', authenticateToken ,async (req, res) => {
//   try {
//     const userId = req.user.userId;
    
//     // Find all history entries for this user and populate challenge details
//     const historyEntries = await History.find({ userId })
//       .populate('challengeId', 'title duration category')
//       .sort({ createdAt: -1 }); // Sort by newest first
      
    
//     // Calculate statistics
//     let totalChallenges = historyEntries.length;
//     let completedChallenges = historyEntries.filter(entry => entry.isCompleted).length;
//     let successRate = totalChallenges > 0 ? Math.round((completedChallenges / totalChallenges) * 100) : 0;
    
//     let ethEarned = historyEntries
//       .filter(entry => entry.isCompleted)
//       .reduce((total, entry) => total + entry.rewardsEarned, 0);
    
//     let ethLost = historyEntries
//       .filter(entry => !entry.isCompleted)
//       .reduce((total, entry) => total + entry.ethStaked, 0);
    
//     // Format the response
//     const response = {
//       stats: {
//         totalChallenges,
//         successRate: `${successRate}%`,
//         ethEarned,
//         ethLost
//       },
//       historyEntries
//     };
    
//     res.status(200).json(response);
//   } catch (error) {
//     console.error('Error fetching user history:', error);
//     res.status(500).json({ message: 'Failed to fetch history data' });
//   }
// });

// Improved API with more detailed challenge information
Historyrouter.get('/user/:token', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Find all history entries for this user with full challenge details
    const historyEntries = await History.find({ userId })
      .populate('challengeId')  // This populates the entire challenge document
      .sort({ createdAt: -1 });
    
    // Calculate statistics
    let totalChallenges = historyEntries.length;
    let completedChallenges = historyEntries.filter(entry => entry.isCompleted).length;
    let successRate = totalChallenges > 0 ? Math.round((completedChallenges / totalChallenges) * 100) : 0;
    
    let ethEarned = historyEntries
      .filter(entry => entry.isCompleted)
      .reduce((total, entry) => total + entry.rewardsEarned, 0);
    
    let ethLost = historyEntries
      .filter(entry => !entry.isCompleted)
      .reduce((total, entry) => total + entry.ethStaked, 0);
    
  
    
    // Format the response
    const response = {
      stats: {
        totalChallenges,
        successRate: `${successRate}%`,
        ethEarned,
        ethLost,
       
      },
      historyEntries: historyEntries.map(entry => ({
        ...entry.toObject(),
        challengeName: entry.challengeId?.name || 'Unknown Challenge',
       
        
      }))
    };
    
    res.status(200).json(response);
  } catch (error) {
    console.error('Error fetching user history:', error);
    res.status(500).json({ message: 'Failed to fetch history data' });
  }
});



module.exports = Historyrouter;