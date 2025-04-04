const express = require("express");

const ActiveChallenge  = require("../Model/ActiveChallege");
const { Challenge } = require("../config/db"); // Ensure Challenge model is imported
const { User } = require("../config/db"); // Ensure User model is imported
const authenticateToken = require("../Middleware/Authenticatetowken"); // Middleware to verify JWT

const ActiveChallengeRouter = express.Router();

ActiveChallengeRouter.post("/create/:token", authenticateToken, async (req, res) => {
  try {

    console.log("hello active challenge")
    console.log(req.user.userId);
    const userId = req.user.userId;
    
    const { challengeId, ethStaked, rewardsEarned, isCompleted } = req.body;


    // Validate if challengeId exists
    const challenge = await Challenge.findById(challengeId);
    if (!challenge) {
      return res.status(404).json({ message: "Challenge not found" });
    }

    
     // ✅ Prepare exercises from Challenge schema
     const exercises = challenge.exercises.map((exercise) => ({
      name: exercise.name,
      reps: exercise.reps,
      completedReps: 0,
      isCompleted: false,
      isVideoRequired: exercise.name.toLowerCase() !== "steps", // ✅ Step doesn't need a video
      videoUrl: "" // ✅ Default empty video URL
    }));

    // Create Active Challenge document
    const newChallenge = new ActiveChallenge({
      userId,
      challengeId,
      exercises, // Use exercises from Challenge schema
      ethStaked,
      rewardsEarned,

      isCompleted,
    });

    await newChallenge.save();
    res.status(201).json({ message: "Active Challenge created successfully!", challenge: newChallenge });
  } catch (error) {
    console.error("Error creating Active Challenge:", error);
    res.status(500).json({ message: "Internal Server Error", error: error.message });
  }
});

ActiveChallengeRouter.get("/get/:token", authenticateToken, async (req, res) => {
  try {
    // console.log("hello")
    const userId = req.user.userId; 

    
    // Find active challenge that's not completed
    const activeChallenge = await ActiveChallenge.findOne({ 
      userId,
      isCompleted: false 
    });

    if (!activeChallenge) {
      return res.status(404).json({ message: "No active challenge found" });
    }

    res.json(activeChallenge);
  } catch (error) {
    console.error("Error fetching active challenge:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

ActiveChallengeRouter.put("/update/:token", authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId; // Get userId from token
    const { exerciseId } = req.body; // Get exerciseId from request body

    // Find the active challenge for the user that is not completed
    const activeChallenge = await ActiveChallenge.findOne({
      userId,
      isCompleted: false,
    });

    if (!activeChallenge) {
      return res.status(404).json({ message: "No active challenge found" });
    }

    // Find the exercise by matching the ID (assuming exerciseId matches the name or another field)
    const exercise = activeChallenge.exercises.find((ex) => ex._id.toString() === exerciseId);

    if (!exercise) {
      return res.status(404).json({ message: "Exercise not found in active challenge" });
    }

    // Update the exercise's completedReps and isCompleted status
    exercise.completedReps = exercise.reps;  // Mark completed reps as the total reps
    exercise.isCompleted = true;  // Mark exercise as completed

    // Save the updated active challenge
    await activeChallenge.save();

    res.status(200).json({ message: "Exercise marked as completed", activeChallenge });
  } catch (error) {
    console.error("Error updating exercise completion:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});












ActiveChallengeRouter.put("/update/challenge/:token", authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;
    

    // Find and update the active challenge
    const updatedChallenge = await ActiveChallenge.findOneAndUpdate(
      { 
        userId,
      },
      { 
        $set: { 
          isCompleted: true,
          
        },
      
      },
      { 
        new: true // Return the updated document
      }
    );

    if (!updatedChallenge) {
      return res.status(404).json({ 
        message: "No active challenge found to update" 
      });
    }

    res.status(200).json({
      message: "Challenge marked as completed",
      challenge: updatedChallenge
    });

  } catch (error) {
    console.error("Error updating challenge:", error);
    res.status(500).json({ 
      message: "Server error", 
      error: error.message 
    });
  }
});


ActiveChallengeRouter.delete("/delete/:token", authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId; // Get userId from the token

    // Find and delete the active challenge associated with the userId
    const deletedChallenge = await ActiveChallenge.findOneAndDelete({ userId });

    if (!deletedChallenge) {
      return res.status(404).json({ message: "No active challenge found for this user" });
    }

    // Send success response with the deleted challenge data
    res.status(200).json({ message: "Active challenge deleted successfully", deletedChallenge });
  } catch (error) {
    console.error("Error deleting active challenge:", error);
    res.status(500).json({ message: "Internal Server Error", error: error.message });
  }
});


module.exports = ActiveChallengeRouter;
