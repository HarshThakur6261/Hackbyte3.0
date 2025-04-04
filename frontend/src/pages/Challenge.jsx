import React, { useContext, useEffect, useState } from "react";
import ConnectWallet from "../components/ConnectWallet";
import DesktopChatbot from "../components/DesktopChatbot";
import "../styles/DesktopHome.css";
import chatbot from "../assets/chatbot.png";
import ChallengeSection from "../components/ChallengeSection";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";
import { FaHistory } from "react-icons/fa";
import axios from "axios";
import { useChallengeContext } from "../context/ChallengeContext";
import { useAuth } from "../context/AuthContext";
import { useNavigate } from "react-router-dom";

const Challenge = () => {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  // const { setChallenges } = useChallengeContext();
  const {logout}=useAuth();
  const navigate=useNavigate();

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
            <ConnectWallet />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Avatar className="h-[63px] p-3.5 w-[63px] border-4 border-[#512E8B] rounded-full bg-[#413359] cursor-pointer hover:opacity-80 transition-opacity">
                  <FaHistory color="white" size={30} />
                </Avatar>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem className="cursor-pointer">
                  previous score
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

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

        <ChallengeSection />
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

export default Challenge;