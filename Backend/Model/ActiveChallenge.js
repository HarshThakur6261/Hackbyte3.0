const mongoose = require("mongoose");

const ActiveChallengeSchema = new mongoose.Schema(
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
    // startTime: {
    //   date: { type: Date, default: Date.now },
    //   time: { type: String, default: () => new Date().toISOString().split("T")[1].split(".")[0] },
    // },
    isCompleted: { type: Boolean, default: false }, // Track challenge completion
  },
  { timestamps: true }
);

const ActiveChallenge = mongoose.model("ActiveChallenge", ActiveChallengeSchema);

module.exports = ActiveChallenge;
