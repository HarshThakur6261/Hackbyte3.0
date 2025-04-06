const express = require('express');
const NotificationRouter = express.Router();
const authenticateToken = require('../Middleware/Authenticatetowken');
const Notification = require('../Model/Notification');

// GET unread notification count
NotificationRouter.get('/unread/count', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    console.log(userId)
    const count = await Notification.countDocuments({ userId, read: false });
    res.status(200).json({ count });
  } catch (error) {
    console.error('Error getting unread count:', error);
    res.status(500).json({ message: 'Server error while fetching unread count' });
  }
});

NotificationRouter.get('/recent', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const notifications = await Notification.find({ userId })
      .sort({ createdAt: -1 })
      .limit(10);
    
    res.status(200).json(notifications);
  } catch (error) {
    console.error('Error fetching recent notifications:', error);
    res.status(500).json({ message: 'Server error while fetching recent notifications' });
  }
});
module.exports = NotificationRouter;