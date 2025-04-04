
import { useLocation, useNavigate } from "react-router-dom";


import React, { useEffect, useRef, useState } from "react";

import ConnectWallet from "../components/ConnectWallet";
import DesktopChatbot from "../components/DesktopChatbot";
import "../styles/DesktopHome.css";
import chatbot from "../assets/chatbot.png";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import { useAuth } from "../context/AuthContext";
import ActiveChallengeCard
 from "../components/ActiveChallengeCard";


import Leaderboard from "../components/Leaderboard";


import { FaHistory } from "react-icons/fa";

import axios from "axios";
import { jwtDecode } from "jwt-decode";
import useFitnessData from "../Utils/useStepCount";

const DesktopHome = () => {
  const navigate = useNavigate();
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const { logout, JwtToken } = useAuth();
  const [isCameraOn, setIsCameraOn] = useState(false);
  const{todaySteps}=useFitnessData();
  console.log(todaySteps);

  const [userData, setuserData] = useState();
  

  const getUserData = async () => {
    try {
      const payload = jwtDecode(JwtToken);
      // console.log("Decoded JWT:", payload);
      const response = await axios.get(`http://localhost:3000/api/users/get/${payload.email}`)
      // console.log(response.data);
      setuserData(response.data);
    } catch (error) {
      console.log(error);

    }
  }
  const handleNavigateToCommunity = () => {
    navigate("/community");
  };
  useEffect(() => {
    if (JwtToken) {
      getUserData();
    }
  }, [JwtToken]);

  // useEffect(() => {
  //   console.log('Step Data:', { todaySteps, 
  //     weeklySteps, 
  //     todayCalories, 
  //     weeklyCalories, });
  // }, [todaySteps, 
  //   weeklySteps, 
  //   todayCalories, 
  //   weeklyCalories,]);


const [videoStream, setVideoStream] = useState(null);
const videoRef = useRef(null);

const handleCameraClick = async () => {
  try {
    if (!isCameraOn) {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user" } // Front camera
      });
      setVideoStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } else {
      // Turn off camera
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        setVideoStream(null);
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
      }
    }
    setIsCameraOn(!isCameraOn);
  } catch (err) {
    console.error("Camera error:", err);
    alert("Could not access camera. Please enable permissions.");
  }
};

// Clean up camera on unmount
useEffect(() => {
  return () => {
    if (videoStream) {
      videoStream.getTracks().forEach(track => track.stop());
    }
  };
}, [videoStream]);

  const chatMessages = [
    {
      sender: "bot",
      text: "Hi there! I'm your StakeFit assistant. How can I help you today?",
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    },
  ];

  const handleLogout = () => {

    logout();
    navigate("/login")
  };
  const handleNavigateToHistory = () => {
    navigate("/history");
  };

  const handleToggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  return (
    <div className="desktop-home">
      <div className="desktop-home__container">
        <header className="desktop-home__header">
          <div className="desktop-home__logo-container">
            <img
              src="https://cdn.builder.io/api/v1/image/assets/TEMP/69e8365158abd202fc7d010edd0471beda6cd6aa?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
              alt="Logo"
              className="desktop-home__logo-image"
            />
            <div className="desktop-home__logo-text">StakeFit</div>
          </div>
          <div className="flex gap-4 items-center">
            <div className="overflow-hidden">
              <ConnectWallet />
            </div>

            <button onClick={handleNavigateToHistory} className="rounded-full">
              <Avatar className="h-[63px]  p-3.5 w-[63px] border-4 border-[#512E8B] rounded-full bg-[#413359] cursor-pointer hover:opacity-80 transition-opacity">
                <FaHistory color="white" size={30} />
              </Avatar>
            </button>
            
           

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Avatar className="h-[63px] w-[63px] cursor-pointer hover:opacity-80 transition-opacity">
                  <AvatarImage src="https://github.com/shadcn.png" alt="User" />
                  <AvatarFallback>US</AvatarFallback>
                </Avatar>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem
                  className="cursor-pointer"
                  onClick={handleLogout}
                >
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        <div className="desktop-home__content">
          <div className="desktop-home__left-section">
            <div className="desktop-home__welcome">
              <div className="desktop-home__title">Welcome To StakeFit</div>
              <div className="desktop-home__subtitle">
                Sweat, hustle, and earn
              </div>
              <button
                className="herobutton"
                onClick={() => navigate("/challenge")}
              >
                Take on next challenge
              </button>{" "}
            </div>
            <div className="gap-5 flex flex-col">
              <div className="desktop-home__stats-card">
                <div className="flex-1 w-full max-md:max-w-full">
                  <h2 className="desktop-section-text ">How it works?</h2>
                  <ul className="mt-2.5 text-base leading-9 text-zinc-400 max-md:max-w-full list-none">
                    <li>Connect your wallet and stake ETH</li>
                    <li>Complete your daily step goal</li>
                    <li>Earn rewards for staying active</li>
                  </ul>
                </div>
                <div className="desktop-home__stats-content"></div>
              </div>
              <div className="desktop-home__stats-card">
                <div className="flex-1 w-full max-md:max-w-full">
                  <h2 className="desktop-section-text ">Fitness Tips</h2>
                  <ul className="mt-2.5 text-base leading-9 text-zinc-400 max-md:max-w-full list-none">
                    <li>
                      Start with 3,000-5,000 steps if you're new to walking
                    </li>
                    <li>Take the stairs instead of elevators.</li>
                    <li>
                      Stay hydrated and stretch post-walk to prevent stiffness.
                    </li>
                  </ul>
                </div>
                <div className="desktop-home__stats-content"></div>
              </div>
            </div>
          </div>

         <ActiveChallengeCard></ActiveChallengeCard>
        </div>
      </div>

      {/* Floating Chatbot Bubble */}
      <div className="chatbot-bubble" onClick={handleToggleChatbot}>
        <img src={chatbot} alt="Chatbot" />
      </div>
      <Leaderboard onClick={handleNavigateToCommunity} />
      {/* Chatbot Component */}
      <DesktopChatbot
        isOpen={isChatbotOpen}
        onClose={() => setIsChatbotOpen(false)}
        messages={chatMessages}
      />
    </div>
  );
};

export default DesktopHome;