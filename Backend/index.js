const express = require("express");
const app = express();
const http = require("http");
const { Server } = require("socket.io");
const jwt = require("jsonwebtoken");
const cors = require("cors");
const bodyParser = require("body-parser");
require("dotenv").config();
require("./config/db");

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

// Create HTTP server
const httpServer = http.createServer(app);

// Configure Socket.io
const io = new Server(httpServer, {
  cors: {
    origin: ["http://localhost:5173", "http://localhost:3000"],
    methods: ["GET", "POST"],
    credentials: true
  }
});

// Store user socket connections
const userSockets = {};

// Import Routes
const userData = require("./Routes/userData.js");
const challengeData = require("./Routes/challengeData");
const ActiveChallengeRouter = require("./Routes/ActivechallengeRouter.js");
const flaskRoutes = require("./Routes/flaskRoutes.js");
const Historyrouter = require("./Routes/History.js");
const communityData = require("./Routes/Community");
const { router: communityChallengeRouter, initializeRouter } = require("./Routes/communityChallengeRouter");
const NotificationRouter = require("./Routes/Notification.js");
const todayChallengeRouter = require("./Routes/todaychallenge.js");

// Initialize the community challenge router with socket.io and userSockets
const initializedCommunityChallengeRouter = initializeRouter(io, userSockets);

// Socket.io connection handler
io.on("connection", (socket) => {
  console.log("New client connected:", socket.id);
  
  // Authenticate user and store their socket connection
  socket.on("authenticate", (token) => {
    try {
      // Verify JWT token and get userId
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      const userId = decoded.userId;
      
      // Store the socket connection for this user
      userSockets[userId] = socket.id;
      socket.userId = userId;
      
      console.log(`User ${userId} authenticated on socket ${socket.id}`);
    } catch (error) {
      console.error("Socket authentication error:", error);
    }
  });
  
  socket.on("disconnect", () => {
    console.log("Client disconnected:", socket.id);
    if (socket.userId) {
      delete userSockets[socket.userId];
    }
  });
});

// Use Routes
app.use("/api/users", userData);
app.use("/api/challenges", challengeData);
app.use("/ActiveChallenge", ActiveChallengeRouter);
app.use("/flask", flaskRoutes);
app.use("/history", Historyrouter);
app.use("/community", communityData);
app.use("/communityChallenge", initializedCommunityChallengeRouter);
app.use("/notifications" , NotificationRouter);
app.use("/todaychallenge" , todayChallengeRouter);


// Start server
const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => {
  console.log(`🚀 Node.js server running on http://localhost:${PORT}`);
});



