
import { useLocation, useNavigate } from "react-router-dom";


import React, { useContext, useEffect, useRef, useState } from "react";

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
import HealthOverview from "../components/HealthOverview";
import { ethers } from "ethers";
import Contractcontext from "../context/contractcontext";
import { toast } from "react-toastify";

const DesktopHome = () => {
  const navigate = useNavigate();
  const [isClaiming, setIsClaiming] = useState(false);

  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const { logout, JwtToken } = useAuth();
  const [isCameraOn, setIsCameraOn] = useState(false);
  const { todaySteps } = useFitnessData();
  const { contract } = useContext(Contractcontext);
  useEffect(() => {
    console.log("1",todaySteps);
  },[todaySteps])

  const [userData, setuserData] = useState();
  // console.log(userData)


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
  useEffect(()=>{
    getUserData();
  },[userData])


  // useEffect(() => {
  //   console.log('Step Data:', { todaySteps, 
  //     weeklySteps, 
  //     todayCalories, 
  //     weeklyCalories, });
  // }, [todaySteps, 
  //   weeklySteps, 
  //   todayCalories, 
  //   weeklyCalories,]);
  const handleNavigateToCommunity = () => {
    navigate("/community");
  };

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
      text: "Hi there! I'm your WeFit assistant. How can I help you today?",
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

  const handleAddRewardsEarned = async (ethReward) => {
    try {
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        return;
      }

      // Make a PUT request to update challengeWon field
      await axios.put(
        `http://localhost:3000/api/users/updateRewardsEarned/${token}`, {
          rewardAmount:ethReward
        }, // Assuming this is the route to increment challengeWon

        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

    } catch (error) {
      console.log(error);

    }
  }
  const handleAddtohistory = async () => {
    try {
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        alert("You need to be logged in to delete the challenge.");
        return;
      }

      // Make the DELETE request using axios
      const response = await axios.post(
        `http://localhost:3000/history/create/${token}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Handle successful response
      console.log("history created  successfully:", response.data);

      toast.success("history created  successfully!");
    } catch (error) {
      console.error("Error history created  challenge:", error);
      alert("An error occurred while history created  .");
    }
  }
  const handleDecreasePoints = async () => {
    try {
      const token = localStorage.getItem("JwtToken");
      
      if (!token) {
        console.error("No JWT token found in localStorage");
        toast.error("Please log in to update points");
        return;
      }
  
      const response = await axios.put(
        `http://localhost:3000/api/users/resetPoints/${token}`,
        {}, // No body needed since we're resetting to zero
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );
  
      if (response.data.success) {
        console.log("Points reset to zero successfully");
        // Optionally update your frontend state here
      } else {
        toast.error(response.data.message || "Failed to reset points");
      }
  
    } catch (error) {
      console.error("Error resetting points:", error);
      toast.error(error.response?.data?.message || "Error resetting points");
    }
  };
  // const handleClaimReward = async () => {
  //   // handleDeleteChallenge();
  //   console.log("hello")
  //   try {
  //     const ethAmount = userData?.points*1/10;
  //     const weiAmount = ethers.parseEther(parseFloat(ethAmount/157669).toFixed(18));
  //     const ethReward=ethAmount/157669
  //     setIsClaiming(true);

  //     if (!contract) {
  //       alert("Contract not connected.");
  //       return;
  //     }

  //    console.log("hello")

  //     const tx = await contract.claimReward({value:weiAmount});
  //     await tx.wait(); // Wait for transaction confirmation

  //     console.log("hello")


  //     toast.success("Rewards claimed successfully!");

  //     handleDecreasePoints();
  //     // handleAddRewardsEarned(ethReward);

  //     handleAddtohistory();



  //     // Optionally refresh the challenge data
  //     // fetchActiveChallenge();

  //   } catch (error) {
  //     console.error("Error claiming reward:", error);
  //     alert("Failed to claim rewards");
  //   } finally {
  //     setIsClaiming(false);
  //   }
  // };
  const handleClaimReward = async () => {
    try {
      setIsClaiming(true);
      
      // Check if userData and points exist
      if (!userData || userData.points === undefined || userData.points === null) {
        alert("User points not available. Please refresh the page.");
        return;
      }
      
      if (!contract) {
        alert("Contract not connected.");
        return;
      }
      
      // Calculate the reward amount with proper checks
      const points = Number(userData.points);
      const ethAmount = points * 0.1; // Simplified from points * 1/10
      
      // Make sure ethAmount is valid before division
      if (isNaN(ethAmount) || ethAmount <= 0) {
        alert("Invalid point value. Cannot claim rewards.");
        return;
      }
      
      const ethReward = ethAmount / 157669;
      
      // Make sure ethReward is a valid number before formatting
      if (isNaN(ethReward) || ethReward <= 0) {
        alert("Invalid reward calculation. Cannot claim rewards.");
        return;
      }
      
      console.log("Points:", points);
      console.log("ETH Amount:", ethAmount);
      console.log("ETH Reward:", ethReward);
      
      // Format the value with proper checks
      const ethValue = parseFloat(ethReward).toFixed(18);
      console.log("ETH Value for parseEther:", ethValue);
      const weiAmount = ethers.parseEther(ethValue);
      
      // Proceed with contract call
      const tx = await contract.claimReward({value: weiAmount});
      await tx.wait();
      
      console.log("Transaction completed");
      
      // Handle success
      toast.success("Rewards claimed successfully!");
      handleDecreasePoints();
      // handleAddtohistory();
      
    } catch (error) {
      console.error("Error claiming reward:", error);
      alert(`Failed to claim rewards: ${error.message}`);
    } finally {
      setIsClaiming(false);
    }
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
            <div className="desktop-home__logo-text">WeFit</div>
          </div>
          <div className="flex gap-4 items-center">
            <div className="overflow-hidden">
              <ConnectWallet />
            </div>
            <DropdownMenu>
      <DropdownMenuTrigger >
      <button className="connect-wallet px-4 py-2 bg-[#512E8B] text-white hover:bg-[#512E8B] hover:text-white rounded-full max-w-[180px] truncate font-mono">
  🔥 {userData?.points}
</button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" className="w-48 bg-[#1A0F2B] border border-[#301F4C] rounded-lg shadow-lg z-50">
        <DropdownMenuItem className="cursor-pointer flex flex-col gap-2 p-2">
          {/* Points Display */}
          <div className="flex items-center justify-between w-full px-2 py-1 rounded bg-[#2B1748]">
            <span className="font-medium text-sm">Your Points:</span>
            <span className="font-mono text-sm text-purple-300">🔥 0</span>
          </div>

          {/* Claim Reward Button */}
          <button
            className="w-full px-3 py-1.5 rounded-full text-sm font-medium bg-gradient-to-r from-purple-600 to-pink-600 text-white "
            onClick={handleClaimReward}
          >
            Claim Reward
          </button>

          {/* Leaderboard Link */}
          <button className="w-full px-3 py-1.5 text-sm text-center text-purple-300  rounded-full ">
            View Leaderboard
          </button>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  



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
                  {/* <Leaderboard onClick={handleNavigateToCommunity} /> */}

                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        <div className="desktop-home__content">
          <div className="desktop-home__left-section">
            <div className="desktop-home__welcome">
              <div className="desktop-home__title">Welcome To WeFit</div>
              <div className="desktop-home__subtitle">
                Sweat, hustle, and earn
              </div>
              <button
                className="herobutton"
                onClick={() => navigate("/challenge")}
              >
Show Recommendations
              </button>{" "}
            </div>
            <div className="flex flex-col flex-1"> {/* Added flex-1 */}
              {/* Added h-full */}
              <HealthOverview className="h-full" /> {/* Pass height prop if needed */}

            </div>
          </div>

          <ActiveChallengeCard getUserData={getUserData}></ActiveChallengeCard>
        </div>
      </div>

      {/* Floating Chatbot Bubble */}
      <div className="chatbot-bubble" onClick={handleToggleChatbot}>
        <img src={chatbot} alt="Chatbot" />
      </div>
      {/* <Leaderboard onClick={handleNavigateToCommunity} /> */}
      <div className="leaderboard-bubble" onClick={handleNavigateToCommunity}>
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width="24" 
    height="24" 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round"
    className="text-white"
  >
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path>
    <circle cx="9" cy="7" r="4"></circle>
    <line x1="22" y1="21" x2="16" y2="21"></line>
    <line x1="19" y1="17" x2="19" y2="21"></line>
    <line x1="19" y1="13" x2="19" y2="15"></line>
  </svg>
</div>
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