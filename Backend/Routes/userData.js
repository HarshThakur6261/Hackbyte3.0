// config/db.js
const mongoose = require('mongoose');
require('dotenv').config();

const URL = process.env.MONGO_URI;

mongoose
  .connect(URL)
  .then(() => {
    console.log("User database connected");
  })
  .catch((e) => {
    console.log("User database connection error", e);
  });

// User model
const userSchema = new mongoose.Schema({
  // Auth & Identity
  googleId: {
    type: String,
    unique: true,
    sparse: true,
    required: true
  },
  email: {
    type: String,
    unique: true,
    required: true,
    trim: true,
    lowercase: true,
    match: [/\S+@\S+\.\S+/, 'is invalid']
  },
  name: {
    type: String,
    trim: true
  },
  walletAddress: {
    type: String,
    index: true,
    default: null,
    // validate: {
    //   validator: function (v) {
    //     return /^0x[a-fA-F0-9]{40}$/.test(v);
    //   },
    //   message: props => `${props.value} is not a valid wallet address!`
    // }
  },

  // Challenges
  challengesWon:{
    type: Number,
    default: 0,
    min: 0
  },
  physicalData: {
    height: {
      type: Number, // in cm
      min: 0
    },
    weight: {
      type: Number, // in kg
      min: 0
    },
    bmi: {
      type: Number,
      min: 0
    },
    age: {
      type: Number,
      min: 0
    },
    gender: {
      type: String,
      enum: ['male', 'female', 'other']
    }
  },

  // Activity
  currentStreak: {
    type: Number,
    default: 0,
    min: 0
  },
  totalEthEarned: {
    type: Number,
    default: 0,
    min: 0
  },
  points:{
    type: Number,
    default: 0,
    min: 0
  },
  isPaid:{
    type:Boolean,
  }

}, {
  timestamps: true
});

const challengeSchema = new mongoose.Schema({
  // Core
  name: {
    type: String,
    required: true
  },
  description: {
    type: String
  },
  type: {
    type: String,
    enum: ["steps", "strength", "combo"],
    required: true
  },

  // Requirements
  stepGoal: {
    type: Number
  },
  exercises: [{
    name: {
      type: String,
      required: true
    },
    reps: {
      type: Number,
      required: true
    }
  }],

  // Crypto Economics
  minStake: {
    type: Number,
    required: true,
    min: 0.0000001
  },
  maxStake: {
    type: Number,
    required: true,
    max: 50
  },
  rewardMultiplier: {
    type: Number,
    default: 1.0,
    min: 1.0
  },

  // Metadata
  difficulty: {
    type: String,
    enum: ["easy", "medium", "hard"],
    required: true
  },
  tags: {
    type: [String],
    default: []
  }
}, {
  timestamps: true
});
const User = mongoose.model('User', userSchema);
const Challenge = mongoose.model('Challenge', challengeSchema);



// Export both models
module.exports = { User, Challenge };