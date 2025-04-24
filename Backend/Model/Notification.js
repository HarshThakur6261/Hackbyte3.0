const mongoose = require('mongoose');

const notificationSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  type: { type: String, required: true }, // 'challenge_invite', etc.
  title: { type: String, required: true },
  description: { type: String, required: true },
  data: { type: Object, default: {} }, // Challenge-specific data
  read: { type: Boolean, default: false },
  responded: { type: Boolean, default: false }, // Whether user has accepted/rejected
  createdAt: { type: Date, default: Date.now }
});

const Notification = mongoose.model('Notification', notificationSchema);
module.exports = Notification;