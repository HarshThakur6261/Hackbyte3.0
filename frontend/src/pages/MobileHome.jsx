import React, { useState } from "react";
import StatsCard from "../components/StatsCard";
import StakeSection from "../components/StakeSection";
import ConnectWallet from "../components/ConnectWallet";
import Chatbot from "../components/Chatbot";
import "../styles/MobileHome.css";
const suffix = "/3k";
const value = "1980";
const MobileHome = () => {
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const [inputMessage, setInputMessage] = useState("");

  const handleToggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && isChatbotOpen) {
      handleSendMessage();
    }
  };

  const [chatMessages, setChatMessages] = useState([
    {
      sender: "bot",
      text: "Hi there! I'm your StakeFit assistant. How can I help you today?",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  const handleSendMessage = () => {
    if (inputMessage.trim() === "") return;
    
    // Add user message
    const newUserMessage = {
      sender: "user",
      text: inputMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setChatMessages([...chatMessages, newUserMessage]);
    setInputMessage(""); // Clear the input
    
    // Simulate bot response after a short delay
    setTimeout(() => {
      const botResponses = [
        "I can help you track your daily steps and rewards!",
        "Would you like to know more about staking and earning rewards?",
        "Remember, completing your daily step goal helps you earn more!",
        "You're doing great with your fitness goals!",
        "Your current staked amount is generating rewards based on your activity."
      ];
      
      const randomResponse = botResponses[Math.floor(Math.random() * botResponses.length)];
      
      const newBotMessage = {
        sender: "bot",
        text: randomResponse,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setChatMessages(prevMessages => [...prevMessages, newBotMessage]);
    }, 1000);
  };
  return (
    <div className="mobile-home">
      <header className="mobile-home__header">
        <img
          src="https://cdn.builder.io/api/v1/image/assets/TEMP/69e8365158abd202fc7d010edd0471beda6cd6aa?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
          alt="Logo"
          className="mobile-home__logo"
        />
        <ConnectWallet />
      </header>

      <h1 className="mobile-home__title">Stats</h1>

      <div className="mobile-home__stats">
        <div className="stats-cardd mobile-home__stats-card">
          <img
            src="https://cdn.builder.io/api/v1/image/assets/TEMP/67f672afe80cde17968bfb27e247f0f43d22657f?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
            alt=""
            className="stats-card__icon"
          />
          <div className="stats-cardd__content">
            <div className="stats-card__label">Steps</div>
            <div className="stats-card__value-container">
              <span className="stats-card__value">{value}</span>
              {suffix && <span className="stats-card__suffix">{suffix}</span>}
            </div>
          </div>
        </div>

        <div className="mobile-home__stats-column">
          <StatsCard
            icon="https://cdn.builder.io/api/v1/image/assets/TEMP/3aba6ae72bc3a8a05ac48bd5020179403ebfe4bc?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
            label="Calories"
            value="1200"
            className="mobile-home__stats-card"
          />
          <StatsCard
            icon="https://cdn.builder.io/api/v1/image/assets/TEMP/dd2685e51dde4835462a564a0304a75fb03bb686?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
            label="Time"
            value="7"
            suffix="hr"
            className="mobile-home__stats-card"
          />
        </div>
      </div>

      <StakeSection />

      <div className="mobile-home__info-box">
        <h2 className="mobile-home__info-title">How it works?</h2>
        <div className="mobile-home__info-content">
          Connect your wallet and stake ETH
          <br />
          Complete your daily step goal
          <br />
          Earn rewards for staying active
        </div>
      </div>

      <div className="mobile-home__chat-bar">
        <img
          src="https://cdn.builder.io/api/v1/image/assets/TEMP/0cafdee42b20fc91416cd612ede1a22e067e02da?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
          alt=""
          className="mobile-home__action-icon"
          onClick={handleToggleChatbot}
        />
        <input
          className="mobile-home__action-input"
          placeholder="Start onChain interaction...."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          onClick={!isChatbotOpen ? handleToggleChatbot : undefined}
        />
       
        <img
          src="https://cdn.builder.io/api/v1/image/assets/TEMP/235096c15e380490658228ad51c7459a1bec2c30?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
          alt=""
          className="mobile-home__action-arrow"
          onClick={handleSendMessage}
        />
      </div>
      
      {/* Chatbot Component */}
      <Chatbot 
        isOpen={isChatbotOpen} 
        onClose={() => setIsChatbotOpen(false)}
        messages={chatMessages}
      />
    </div>
  );
};

export default MobileHome;
