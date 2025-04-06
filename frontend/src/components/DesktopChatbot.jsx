import React, { useState, useRef, useEffect } from "react";
import "../styles/DesktopChatbot.css";

const DesktopChatbot = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hi there! I'm your StakeFit assistant. How can I help you today?",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSendMessage = () => {
    if (inputMessage.trim() === "") return;
    
    // Add user message
    const newUserMessage = {
      sender: "user",
      text: inputMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setMessages([...messages, newUserMessage]);
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
      
      setMessages(prevMessages => [...prevMessages, newBotMessage]);
    }, 1000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className={`desktop-chatbot ${isOpen ? 'open' : ''}`}>
      <div className="desktop-chatbot-header">
        <div className="desktop-chatbot-title">StakeFit Assistant</div>
        <button className="desktop-chatbot-close-btn" onClick={onClose}>×</button>
      </div>
      
      <div className="desktop-chatbot-messages">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`desktop-message ${message.sender === "user" ? "user-message" : "bot-message"}`}
          >
            <div className="desktop-message-content">
              <p>{message.text}</p>
              <span className="desktop-message-timestamp">{message.timestamp}</span>
            </div>
            
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="desktop-chatbot-input-container">
        <input
          type="text"
          className="desktop-chatbot-input"
          placeholder="Type your message..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button 
          className="desktop-chatbot-send-btn"
          onClick={handleSendMessage}
        >
          <img 
            src="https://cdn.builder.io/api/v1/image/assets/TEMP/235096c15e380490658228ad51c7459a1bec2c30?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
            alt="Send"
          />
        </button>
      </div>
    </div>
  );
};

export default DesktopChatbot;
