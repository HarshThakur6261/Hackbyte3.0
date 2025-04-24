
const mongoose = require('mongoose');

const communityChallengeSchema = new mongoose.Schema({
  communityId: { type: mongoose.Schema.Types.ObjectId, ref: 'Community', required: true },
  name: { type: String, required: true, trim: true },
  description: { type: String, required: true },
  exercises: [{
    name: String,
    reps: String,
    sets: String,
    duration: String,
    description: String
  }],
  stake: {
    amount: { type: String, required: true, min: 0 },
  },
  createdAt: { type: Date, default: Date.now },

  participants: [{
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    accepted: { type: Boolean, default: false },
    completed: { type: Boolean, default: false },
    completedAt: Date,
    stakeSubmitted: { type: Boolean, default: false }
  }]
});

const communityChallenge = mongoose.model('communityChallenge', communityChallengeSchema);
module.exports = communityChallenge;