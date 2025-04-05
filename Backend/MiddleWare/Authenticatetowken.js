const jwt = require("jsonwebtoken");

const authenticateToken = (req, res, next) => {
  const token = req.params.token; // Extract token from request params

  if (!token) {
    return res.status(401).json({ message: "Access Denied. No Token Provided." });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET); // Replace with your actual secret key
    req.user = decoded; // Attach user details to request
    
    next();
  } catch (error) {
    return res.status(403).json({ message: "Invalid Token", error: error.message });
  }
};

module.exports = authenticateToken;
