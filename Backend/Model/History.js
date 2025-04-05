const mongoose = require("mongoose");

const HistorySchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    challengeId: { type: mongoose.Schema.Types.ObjectId, ref: "Challenge", required: true },
    exercises: [
      {
        name: { type: String, required: true },
        reps: { type: Number, required: true },
        completedReps: { type: Number, default: 0 },
        isCompleted: { type: Boolean, default: false },
        isVideoRequired: { type: Boolean, default: false }, // ✅ New field
        videoUrl: { type: String, default: "" } // ✅ New field for uploaded video
      }
    ],
    ethStaked: { type: Number, required: true },
    rewardsEarned: { type: Number, default: 0 },
   
    isCompleted: { type: Boolean, default: false }, // Track challenge completion
  },
  { timestamps: true }
);

const History = mongoose.model("History", HistorySchema);

module.exports = History;
