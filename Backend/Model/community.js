const mongoose = require('mongoose');

const communitySchema = new mongoose.Schema({
  leader: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  description: { type: String, required: true },
  tags: [{ type: String, trim: true }],
  members: [{
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    joinedAt: { type: Date, default: Date.now },
    role: { type: String, enum: ['leader', 'member'], default: 'member' },
    points: { type: Number, default: 0 }  // Points for each member
  }],
  todayChallenge: { type: mongoose.Schema.Types.ObjectId, ref: 'Challenge', default: null },
  isPrivate: { type: Boolean, default: false },
//   accessCode: { type: String, default: null },
  completionPercentage: { type: Number, default: 0 },
  createdAt: { type: Date, default: Date.now }
});

const Community = mongoose.model('Community', communitySchema);

module.exports = Community;
