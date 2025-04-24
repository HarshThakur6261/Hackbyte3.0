const express = require('express');
const jwt = require('jsonwebtoken');
const { User } = require('../config/db'); // Adjust path as needed
const authenticateToken = require('../Middleware/Authenticatetowken');


const Community = require('../Model/community');

const communityData = express.Router();

// Delete community
communityData.delete('/:id/:token', async (req, res) => {
  try {
    // Verify token and user permissions
    const token = req.params.token;
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const userId = decoded.id;
    
    // Find community
    const community = await Community.findById(req.params.id);
    
    if (!community) {
      return res.status(404).json({ message: 'Community not found' });
    }
    
    // Check if user is a leader
    const userMember = community.members.find(
      member => member.userId.toString() === userId.toString()
    );
    
    if (!userMember || userMember.role !== 'leader') {
      return res.status(403).json({ message: 'Only community leaders can delete the community' });
    }
    
    // Delete the community
    await Community.findByIdAndDelete(req.params.id);
    
    // Remove community reference from all users
    await User.updateMany(
      { communities: req.params.id },
      { $pull: { communities: req.params.id } }
    );
    
    res.json({ message: 'Community deleted successfully' });
  } catch (error) {
    console.log(error);
    
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ message: 'Invalid token' });
    }
    
    res.status(500).json({ message: error.message });
  }
});

// Delete community
communityData.get('/new', async (req, res) => {
  try {
    console.log("Fetching communities...");

    const communities = await Community.find()

    console.log("Communities found:", communities);

    if (communities.length === 0) {
      return res.status(404).json({ message: 'No communities found' });
    }

    res.json(communities);
  } catch (error) {
    console.error("Error in /community/available:", error);

    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ message: 'Invalid token' });
    }

    res.status(500).json({ message: error.message });
  }
});


// Get all communities for a specific user
communityData.get('/all/:token', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Find communities where the user is a member
    const communities = await Community.find({
      'members.userId': userId
    });
    
    res.status(200).json(communities);
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: error.message });
  }
});


communityData.get('/users/:token', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;
    console.log("User ID:ff", userId);
    
    // Execute the query with await and get the actual user document
    const user = await User.findById(userId);
    
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    // Return the user document (this will be automatically converted to JSON)
    res.status(200).json(user);
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: error.message });
  }
});

// Create a new community
communityData.post('/create/:token',authenticateToken, async (req, res) => {
  try {
    // Extract token and verify
    // const token = req.params.token;
    // const decoded = jwt.verify(token, process.env.JWT_SECRET);
    // const userId = decoded.id;
    const userId = req.user.userId; 
    
    // Extract data from request body
    const { name, description, tags} = req.body;
    
    // Validate required fields
    if (!name || !description) {
      return res.status(400).json({ message: 'Name and description are required' });
    }
    
    // If community is private, access code is required
    
    
    // Create new community
    const community = new Community({
      leader:userId,
      name,
      description,
      tags: tags || [],
      isPrivate:false,
    //   accessCode: isPrivate ? accessCode : null,
      members: [{
        userId,
        role: 'leader',
        joinedAt: Date.now()
      }]
    });
    console.log(community)
    // Save to database
    const newCommunity = await community.save();
    
   
    
    res.status(201).json(newCommunity);
  } catch (error) {
    console.log(error);
    
    // Handle JWT errors
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ message: 'Invalid token' });
    }
    
    // Handle duplicate key errors
    if (error.code === 11000) {
      return res.status(409).json({ message: 'Community with this name already exists' });
    }
    
    res.status(400).json({ message: error.message });
  }
});

communityData.post('/join/:token', authenticateToken, async (req, res) => {
  try {
    // Extract user ID from authenticated token
    const userId = req.user.userId;
    
    // Get community ID from URL params
    const { communityId } = req.body;
    
    // Optional role parameter (default to 'member')
    const { role = 'member' } = req.body;
    
    // Validate role (only allow 'member' unless authorized differently)
    if (role !== 'member' && role !== 'moderator') {
      return res.status(400).json({ message: 'Invalid role specified' });
    }
    
    // Find the community
    const community = await Community.findById(communityId);
    
    // Check if community exists
    if (!community) {
      return res.status(404).json({ message: 'Community not found' });
    }
    
    // Check if user is already a member
    const existingMember = community.members.find(member => 
      member.userId.toString() === userId.toString()
    );
    
    if (existingMember) {
      return res.status(409).json({ message: 'User is already a member of this community' });
    }
    
    
    
    // Add user to community members
    community.members.push({
      userId,
      role: role, // Default is 'member' as set above
      joinedAt: Date.now()
    });
    
    // Save updated community
    await community.save();
    
    // Return success response with limited community data
    res.status(200).json({
      message: 'Successfully joined community',
      community: {
        id: community._id,
        name: community.name,
        role: role
      }
    });
    
  } catch (error) {
    console.log(error);
    
    // Handle JWT errors
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ message: 'Invalid token' });
    }
    
    res.status(500).json({ message: 'Error joining community', error: error.message });
  }
});

// Get user's communities
communityData.get('/user/:token', async (req, res) => {
  try {
    const token = req.params.token;
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const userId = decoded.id;
    
    const communities = await Community.find({
      'members.userId': userId
    }).select('name description tags isPrivate members createdAt');
    
    // Format data for frontend
    const userCommunities = communities.map(community => {
      const userMember = community.members.find(
        member => member.userId.toString() === userId.toString()
      );
      
      return {
        _id: community._id,
        name: community.name,
        description: community.description,
        tags: community.tags,
        isPrivate: community.isPrivate,
        role: userMember.role,
        joinedAt: userMember.joinedAt,
        points: userMember.points || 0,
        membersCount: community.members.length,
        createdAt: community.createdAt
      };
    });
    
    res.status(200).json(userCommunities);
  } catch (error) {
    console.log(error);
    
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ message: 'Invalid token' });
    }
    
    res.status(500).json({ message: error.message });
  }
});

// Get community by ID
communityData.get('/:id', async (req, res) => {
  try {
    const community = await Community.findById(req.params.id);
    if (!community) return res.status(404).json({ message: 'Community not found' });
    res.json(community);
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: error.message });
  }
});

// Update community
communityData.patch('/:id/:token', async (req, res) => {
  try {
    // Verify token and user permissions
    const token = req.params.token;
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const userId = decoded.id;
    
    // Find community
    const community = await Community.findById(req.params.id);
    
    if (!community) {
      return res.status(404).json({ message: 'Community not found' });
    }
    
    // Check if user is a leader
    const userMember = community.members.find(
      member => member.userId.toString() === userId.toString()
    );
    
    if (!userMember || userMember.role !== 'leader') {
      return res.status(403).json({ message: 'Only community leaders can update community details' });
    }
    
    // Update community
    const updatedCommunity = await Community.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    );
    
    res.json(updatedCommunity);
  } catch (error) {
    console.log(error);
    
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ message: 'Invalid token' });
    }
    
    res.status(400).json({ message: error.message });
  }
});

communityData.get('/:communityId/members', async (req, res) => {
  try {
    const { communityId } = req.params;

    const community = await Community.findById(communityId)
      .populate('members.userId', 'name email profilePicture'); // populate only selected fields (optional)

    if (!community) {
      return res.status(404).json({ error: 'Community not found' });
    }

    res.json(community.members);
  } catch (error) {
    console.error('Error fetching community members:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

communityData.get('/community/:communityId/leaderboard', authenticateToken, async (req, res) => {
  try {
    const { communityId } = req.params;

    // Get community members
    const community = await Community.findById(communityId)
      .populate({
        path: 'members.userId',
        select: 'name walletAddress points'
      });

    if (!community) {
      return res.status(404).json({ message: 'Community not found' });
    }

    // Sort members by points (descending)
    const sortedMembers = community.members
      .map(member => ({
        userId: member.userId,
        points: member.userId.points || 0
      }))
      .sort((a, b) => b.points - a.points);

    res.status(200).json({
      success: true,
      leaderboard: sortedMembers
    });
  } catch (error) {
    console.error('Error fetching leaderboard:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
});

// Update user points
communityData.put('/users/:userId/points', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    const { points } = req.body;

    // Verify the requesting user has permission
    if (req.user.userId !== userId) {
      return res.status(403).json({ message: 'Unauthorized' });
    }

    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { $set: { points } },
      { new: true, select: 'name points' }
    );

    if (!updatedUser) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.status(200).json({
      success: true,
      user: updatedUser
    });
  } catch (error) {
    console.error('Error updating points:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
});







module.exports = communityData;