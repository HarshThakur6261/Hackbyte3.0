const express = require("express");
const app = express();
require("./config/db");
const cors = require("cors");
const bodyParser = require("body-parser");

const { createServer } = require("http");
const { Server } = require("socket.io");

require("dotenv").config();

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());



// Import Routes
const userData = require("./Routes/userData.js");
const challengeData = require("./Routes/challengeData");
const ActiveChallengeRouter = require("./Routes/ActiveChallengeRouter");
const Historyrouter = require("./Routes/History");


// Use Routes
app.use("/api/users", userData);
app.use("/api/challenges", challengeData);
app.use("/ActiveChallenge", ActiveChallengeRouter);

app.use("/history" , Historyrouter)






// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {

  console.log(`🚀 Node.js server running on http://localhost:${PORT}`);
});
