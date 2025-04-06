## 🚀 Deployment Guide

Here’s how to run *FitnessConnect* on your local system.

### 📁 Project Structure


fitnessconnect/
├── client/           # React frontend (Vite)
├── server/           # Node.js backend
├── flask_backend/    # Flask backend (Pose Detection & Recommendation)
└── smart_contracts/  # Solidity Smart Contracts


---

### 🔧 1. Run the React Frontend

bash
cd client
npm install
npm run dev


💻 Frontend URL:  
*http://localhost:5173/*

---

### 🔧 2. Run the Node.js Backend

bash
cd server
npm install
npm start


This backend handles:
- User authentication (JWT, OAuth)
- Challenge creation and tracking
- MongoDB data management
- Smart contract interaction

---

### 🔧 3. Run the Flask Backend (ML & Pose Detection)

bash
cd flask_backend
pip install -r requirements.txt
python app.py


This Flask server powers:
- Exercise posture detection (via Mediapipe)
- Video analysis (OpenCV, TensorFlow)
- Personalized recommendation engine (Hybrid ML model)

🧠 Backend API URL:  
*http://127.0.0.1:5000/*

---

### 🌐 API Overview

- *Frontend*: http://localhost:5173/
- *Node.js Backend*: Runs on default Express port (e.g. 3000)
- *Flask Backend*: http://127.0.0.1:5000/

Make sure:
- MongoDB is connected and configured.
- MetaMask is connected to Sepolia Testnet.
- Smart contracts are deployed via Remix IDE.

---

## 📜 License

This project is licensed under the MIT License.
