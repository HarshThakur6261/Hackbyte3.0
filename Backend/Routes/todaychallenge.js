const express = require('express');
const Community = require('../Model/community');
const todayChallengeRouter = express.Router();


// GET unread notification count
// GET today's challenge for a community
todayChallengeRouter.get('/:communityId', async (req, res) => {
  try {
    console.log("enterrrrrrrrrrrrrrrr")
    const { communityId } = req.params;
    
    // Find the community and populate today's challenge
    const community = await Community.findById(communityId)
      .populate('todayChallenge')
      .exec();
    console.log(community);
    if (!community) {
      return res.status(404).json({ message: 'Community not found' });
    }
    
    // Check if there's a today's challenge set
    if (!community.todayChallenge) {
      return res.status(404).json({ message: 'No active challenge found for today' });
    }
    
    // Return just the challenge data
    res.status(200).json(community.todayChallenge);
    
  } catch (error) {
    console.error('Error fetching today\'s challenge:', error);
    res.status(500).json({ message: 'Server error while fetching today\'s challenge' });
  }
});


module.exports = todayChallengeRouter;