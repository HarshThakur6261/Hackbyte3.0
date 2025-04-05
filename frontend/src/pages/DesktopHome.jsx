import { useNavigate } from "react-router-dom";

import React, { useEffect, useRef, useState } from "react";

import HealthOverview from "../components/HealthOverview";
import UserHeader from "../components/UserHeader";
import ActivityTracker from "../components/ActivityTracker";
import ChallengeTracker from "../components/ChallengeTracker";
import ConnectWallet from "../components/ConnectWallet";
import DesktopChatbot from "../components/DesktopChatbot";
import "../styles/DesktopHome.css";
import chatbot from "../assets/chatbot.png";
import Leaderboard from "../components/Leaderboard"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import { useAuth } from "../context/AuthContext";
import ActiveChallengeCard from "../components/ActiveChallengeCard";

import { FaHistory } from "react-icons/fa";

import axios from "axios";
import { jwtDecode } from "jwt-decode";

const DesktopHome = () => {
  const navigate = useNavigate();
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const { logout, JwtToken } = useAuth();
  const [isCameraOn, setIsCameraOn] = useState(false);

  const [userData, setuserData] = useState();

  const getUserData = async () => {
    try {
      const payload = jwtDecode(JwtToken);
      // console.log("Decoded JWT:", payload);
      const response = await axios.get(
        `http://localhost:3000/api/users/get/${payload.email}`
      );
      // console.log(response.data);
      setuserData(response.data);
    } catch (error) {
      console.log(error);
    }
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
          video: { facingMode: "user" }, // Front camera
        });
        setVideoStream(stream);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } else {
        // Turn off camera
        if (videoStream) {
          videoStream.getTracks().forEach((track) => track.stop());
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
        videoStream.getTracks().forEach((track) => track.stop());
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
    navigate("/login");
  };

  const handleToggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };
  const handleNavigateToHistory = () => {
    navigate("/history");
  };
  const handleNavigateToCommunity = () => {
    navigate("/community");
  };
  const handleProfile = () => {
    navigate("/input");
  };

  return (
    <main className="flex overflow-hidden relative flex-col justify-center items-center px-20  min-h-[992px] max-md:px-5">
          <img
            src="https://cdn.builder.io/api/v1/image/assets/TEMP/57cec06690767ebb67f7c45d0656b40cae90d486?placeholderIfAbsent=true&apiKey=3cb565296e9348b38bc8cd244ca00b7e"
            className="object-cover absolute inset-0 size-full"
            alt="Background"
          />
          <div className="relative max-w-full w-[1171px]">
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
                <button onClick={handleNavigateToCommunity} className="rounded-full">
                  <Avatar className="h-[63px]  p-3.5 w-[63px] border-4 border-[#512E8B] rounded-full bg-[#350091] cursor-pointer hover:opacity-80 transition-opacity">
                    <FaHistory color="white" size={30} />
                  </Avatar>
                </button>
    
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
                      onClick={handleProfile}
                    >
                      Profile
                    </DropdownMenuItem>
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
            <section className=" w-full max-md:mt-10 max-md:max-w-full">
              <UserHeader />
              <div className="flex flex-wrap gap-9 items-center mt-10 w-full max-md:max-w-full">
                <ActivityTracker />
                <ChallengeTracker />
              </div>
            </section>
          </div>
        </main>
  );
};

export default DesktopHome;
