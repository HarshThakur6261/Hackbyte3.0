const express = require('express');
const { Challenge } = require('../config/db');

const challengeData = express.Router();

challengeData.get('/get', async (req, res) => {
  try {
    console.log("running")
    const challenges = await Challenge.find();
    res.status(201).json(challenges);

  } catch (error) {
    console, log(error);
    res.status(404);
  }
})

challengeData.post('/', async (req, res) => {
  const challenge = new Challenge({
    name: req.body.name,
    type: req.body.type,
    stepGoal: req.body.stepGoal,
    exercises: req.body.exercises,
    minStake: req.body.minStake,
    maxStake: req.body.maxStake,
    rewardMultiplier: req.body.rewardMultiplier,
    difficulty: req.body.difficulty,
    tags: req.body.tags || []
  });

  try {
    const newChallenge = await challenge.save();
    res.status(201).json(newChallenge);
  } catch (err) {
    res.status(400).json({ message: err.message });
  }
});


// Get challenge by ID
challengeData.get('/:id', async (req, res) => {
  try {
    const challenge = await Challenge.findById(req.params.id);
    if (!challenge) return res.status(404).json({ message: 'Challenge not found' });
    res.json(challenge);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// Update challenge
challengeData.patch('/:id', async (req, res) => {
  try {
    const challenge = await Challenge.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    );
    res.json(challenge);
  } catch (err) {
    res.status(400).json({ message: err.message });
  }
});

// Delete challenge
challengeData.delete('/:id', async (req, res) => {
  try {
    await Challenge.findByIdAndDelete(req.params.id);
    res.json({ message: 'Challenge deleted' });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

module.exports = challengeData;

