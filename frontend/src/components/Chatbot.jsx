import React, { useRef, useEffect } from "react";
import "../styles/Chatbot.css";

const Chatbot = ({ isOpen, onClose, messages }) => {
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div className={`chatbot-container ${isOpen ? 'open' : ''}`}>
      <div className="chatbot-header">
        <div className="chatbot-title">WeFit Assistant</div>
        <button className="chatbot-close-btn" onClick={onClose}>×</button>
      </div>
      
      <div className="chatbot-messages">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`message ${message.sender === "user" ? "user-message" : "bot-message"}`}
          >
            <div className="message-content">
              <p>{message.text}</p>
              <span className="message-timestamp">{message.timestamp}</span>
            </div>
            
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      {/* We're removing the chatbot input container since we're using the main input from MobileHome */}
    </div>
  );
};

export default Chatbot;
