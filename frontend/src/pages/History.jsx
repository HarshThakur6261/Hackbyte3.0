import { useNavigate } from "react-router-dom";

import React, { useEffect, useRef, useState } from "react";
import ConnectWallet from "../components/ConnectWallet";
import DesktopChatbot from "../components/DesktopChatbot";
import "../styles/DesktopHome.css";
import StatisticCard from "../components/StatisticCard";
import ChallengeCard from "../components/HistoryChallengeCard";
import chatbot from "../assets/chatbot.png";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import { useAuth } from "../context/AuthContext";

import { FaHistory } from "react-icons/fa";

import axios from "axios";
import { jwtDecode } from "jwt-decode";

const History = () => {
  const navigate = useNavigate();
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const { logout, JwtToken } = useAuth();

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
  const handleNavigateToHistory = () => {
    navigate("/history");
  };

  const handleToggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  return (
    <div className="desktop-home">
      <div className="desktop-history__container">
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

        <main className="flex flex-col gap-10 mx-auto w-full ">
          <header className="flex justify-between items-center w-full max-md:flex-col max-md:gap-5">
            <h1 className="text-3xl font-medium text-white">
              Challenge History
            </h1>
            <div className="flex gap-6 items-center">
              <select className="w-[180px] bg-[#413359] h-10 pl-2 pr-2  py-2 rounded-md  border-input  text-sm text-white">
                <option value="" disabled selected>All Challenges</option>
                <option value="apple">Apple</option>
                <option value="banana">Banana</option>
                <option value="blueberry">Blueberry</option>
                <option value="grapes">Grapes</option>
                <option value="pineapple">Pineapple</option>
              </select>              
              <select className="w-[180px] bg-[#413359] h-10 pl-2 pr-2  py-2 rounded-md  border-input  text-sm text-white">
                <option value="" disabled selected>Newest</option>
                <option value="apple">Apple</option>
                <option value="banana">Banana</option>
                <option value="blueberry">Blueberry</option>
                <option value="grapes">Grapes</option>
                <option value="pineapple">Pineapple</option>
              </select>             </div>
          </header>

          <section className="flex flex-col gap-11">
            <div className="flex gap-8 items-center w-full max-md:flex-wrap max-md:justify-center max-sm:flex-col">
              <StatisticCard
                title="Total Challenges"
                value="156"
                textColor="text-white"
              />
              <StatisticCard
                title="Success Rate"
                value="94%"
                textColor="text-white"
              />
              <StatisticCard
                title="ETH Earned"
                value="+ 2.45 ETH"
                textColor="text-green-500"
              />
              <StatisticCard
                title="ETH Lost"
                value="-0.13 ETH"
                textColor="text-red-600"
              />
            </div>

            <section className="flex flex-wrap gap-9 justify-center">
              <ChallengeCard
                title="Morning HIIT"
                date="2024-01-15"
                amount="+ 0.05 ETH"
                status="Completed"
                progress="20/20"
                isSuccess={true}
              />
              <ChallengeCard
                title="Morning HIIT"
                date="2024-01-15"
                amount="+ 0.05 ETH"
                status="Completed"
                progress="20/20"
                isSuccess={true}
              />
              <ChallengeCard
                title="Morning HIIT"
                date="2024-01-15"
                amount="+ 0.05 ETH"
                status="Completed"
                progress="20/20"
                isSuccess={true}
              />
              <ChallengeCard
                title="Core Focus"
                date="2024-02-15"
                amount="- 0.05 ETH"
                status="Incomplete"
                progress="10/20"
                isSuccess={false}
              />
              <ChallengeCard
                title="Morning HIIT"
                date="2024-01-15"
                amount="+ 0.05 ETH"
                status="Completed"
                progress="20/20"
                isSuccess={true}
              />
              <ChallengeCard
                title="Morning HIIT"
                date="2024-01-15"
                amount="+ 0.05 ETH"
                status="Completed"
                progress="20/20"
                isSuccess={false}
              />
            </section>
          </section>
        </main>
      </div>

      {/* Floating Chatbot Bubble */}
      <div className="chatbot-bubble" onClick={handleToggleChatbot}>
        <img src={chatbot} alt="Chatbot" />
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

export default History;
