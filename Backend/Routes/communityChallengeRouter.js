const express = require('express');
const router = express.Router();
const CommunityChallenge = require('../Model/communityChallenge');
const communityChallenge = require('../Model/communityChallenge');
const Community = require('../Model/community');
const authenticateToken = require('../Middleware/Authenticatetowken');
const Notification = require('../Model/Notification');

// Reference to userSockets and io - will be set when router is initialized
let userSockets;
let io;

// Initialize router with socketIO and userSockets references
const initializeRouter = (socketIO, socketUsers) => {
  io = socketIO;
  userSockets = socketUsers;
  return router;
};

// ✅ Create a new community challenge
router.post('/create', async (req, res) => {
  try {
    console.log("backend", req.body);
    const { communityId, name, description, exercises, stake } = req.body;
   
    // Create the challenge
    const challenge = new communityChallenge({
      communityId,
      name,
      description,
      exercises,
      stake,
      participants: []
    });
    
    const savedChallenge = await challenge.save();
    console.log(savedChallenge);
    // Get all users in the community
    const community = await Community.findById(communityId);
    console.log(community);
    if (!community) {
      return res.status(404).json({ message: "Community not found" });
    }
    
    // Create notifications for all community members
    const memberIds = community.members.map(member => member.userId);
    console.log(memberIds);
    
    // Create notifications array
    const notifications = memberIds.map(userId => {
      return {
        userId,
        type: 'challenge_invite',
        title: `New Challenge: ${name}`,
        description: `You've been invited to join "${name}" challenge`,
        data: {
          challengeId: savedChallenge._id,
          communityId: communityId,
          stakeAmount: stake.amount,
          challengeName: name,
          challengeDescription: description
        },
        read: false,
        createdAt: new Date()
      };
    });
    
    // Save notifications to database
    const savedNotifications = await Notification.insertMany(notifications);
    
    // Send real-time notifications to connected users
    if (io && userSockets) {
      savedNotifications.forEach(notification => {
        const socketId = userSockets[notification.userId.toString()];
        if (socketId) {
          io.to(socketId).emit('new_notification', notification);
        }
      });
    }
    
    res.status(200).json(savedChallenge);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: err.message });
  }
});

// 📥 Get all challenges for a community
router.get('/community/:communityId', async (req, res) => {
  try {
    const challenges = await CommunityChallenge.find({ communityId: req.params.communityId });
    res.json(challenges);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 👤 Join a challenge
router.post('/join/:challengeId', async (req, res) => {
  const { userId } = req.body;
  try {
    const challenge = await CommunityChallenge.findById(req.params.challengeId);
    if (!challenge) return res.status(404).json({ error: 'Challenge not found' });

    const alreadyParticipant = challenge.participants.find(p => p.userId.toString() === userId);
    if (alreadyParticipant) return res.status(400).json({ error: 'User already joined' });

    challenge.participants.push({ userId });
    await challenge.save();

    res.json({ message: 'Joined challenge successfully', challenge });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ✅ Accept challenge
router.post('/accept/:challengeId', async (req, res) => {
  const { userId } = req.body;
  try {
    const challenge = await CommunityChallenge.findById(req.params.challengeId);
    const participant = challenge.participants.find(p => p.userId.toString() === userId);

    if (!participant) return res.status(404).json({ error: 'User not in participant list' });

    participant.accepted = true;
    participant.stakeSubmitted = true; // Assume stake is auto-submitted
    await challenge.save();

    res.json({ message: 'Challenge accepted', challenge });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ✅ Mark challenge as completed
router.post('/complete/:challengeId', async (req, res) => {
  const { userId } = req.body;
  try {
    const challenge = await CommunityChallenge.findById(req.params.challengeId);
    const participant = challenge.participants.find(p => p.userId.toString() === userId);

    if (!participant) return res.status(404).json({ error: 'User not in participant list' });

    participant.completed = true;
    participant.completedAt = new Date();
    await challenge.save();

    res.json({ message: 'Challenge marked as completed', challenge });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});
router.get('/community/:communityId/leaderboard', authenticateToken, async (req, res) => {
  try {
    const { communityId } = req.params;

    // 1. Get all completed challenges for this community
    const challenges = await CommunityChallenge.find({
      communityId,
      'participants.completed': true
    }).lean();

    // 2. Aggregate user points and challenge completions
    const userStats = {};

    challenges.forEach(challenge => {
      challenge.participants.forEach(participant => {
        if (participant.completed) {
          const userId = participant.userId.toString();
          
          if (!userStats[userId]) {
            userStats[userId] = {
              userId,
              points: 0,
              completedChallenges: 0,
              lastCompleted: null
            };
          }

          userStats[userId].points += challenge.rewardPoints || 10; // Default 10 points per challenge
          userStats[userId].completedChallenges += 1;
          userStats[userId].lastCompleted = 
            new Date(Math.max(
              new Date(participant.completedAt || challenge.updatedAt),
              new Date(userStats[userId].lastCompleted || 0)
            ));
        }
      });
    });

    // 3. Get user details for the leaderboard
    const userIds = Object.keys(userStats);
    const users = await User.find(
      { _id: { $in: userIds } },
      'name walletAddress avatar points'
    ).lean();

    // 4. Combine data for leaderboard
    const leaderboard = users.map(user => {
      const stats = userStats[user._id.toString()] || {
        points: 0,
        completedChallenges: 0
      };
      
      return {
        userId: {
          _id: user._id,
          name: user.name,
          walletAddress: user.walletAddress,
          avatar: user.avatar
        },
        points: stats.points + (user.points || 0), // Combine challenge points with base points
        completedChallenges: stats.completedChallenges,
        lastActivity: stats.lastCompleted
      };
    });

    // 5. Sort by points (descending)
    leaderboard.sort((a, b) => b.points - a.points);

    res.status(200).json({
      success: true,
      leaderboard
    });

  } catch (error) {
    console.error('Error fetching leaderboard:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
});

module.exports = { router, initializeRouter };