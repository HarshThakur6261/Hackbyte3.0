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
  const [isLoading, setIsLoading] = useState(false);
  const [authStatus, setAuthStatus] = useState({
    isLoggedIn: false,
    tokenSource: null,
    tokenPreview: null
  });
  const messagesEndRef = useRef(null);

  // Debug function to check all storage locations
  const debugTokenStorage = () => {
    console.log("=== TOKEN STORAGE DEBUG ===");
    
    // Check localStorage
    console.log("localStorage keys:", Object.keys(localStorage));
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      const value = localStorage.getItem(key);
      console.log(`localStorage[${key}] = ${value ? value.substring(0, 15) + '...' : 'null'}`);
    }
    
    // Check sessionStorage
    console.log("sessionStorage keys:", Object.keys(sessionStorage));
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      const value = sessionStorage.getItem(key);
      console.log(`sessionStorage[${key}] = ${value ? value.substring(0, 15) + '...' : 'null'}`);
    }
    
    // Check cookies
    console.log("cookies:", document.cookie);
    
    console.log("=== END DEBUG ===");
  };

  // Check authentication status
  useEffect(() => {
    const checkAuth = () => {
      // Debug all storage
      debugTokenStorage();
      
      // Try to find token in various storage locations and with various keys
      const tokenKeys = ['token', 'userToken', 'authToken', 'jwt', 'accessToken', 'user'];
      
      // Check localStorage
      for (const key of tokenKeys) {
        const value = localStorage.getItem(key);
        if (value) {
          try {
            // If the value is a JSON object (like a user object with a token property)
            const parsed = JSON.parse(value);
            if (parsed && parsed.token) {
              console.log(`Found token in localStorage.${key}.token`);
              setAuthStatus({
                isLoggedIn: true,
                tokenSource: `localStorage.${key}.token`,
                tokenPreview: parsed.token.substring(0, 10) + '...'
              });
              return parsed.token;
            }
          } catch (e) {
            // Not JSON, treat as direct token
            console.log(`Found potential token in localStorage.${key}`);
            setAuthStatus({
              isLoggedIn: true,
              tokenSource: `localStorage.${key}`,
              tokenPreview: value.substring(0, 10) + '...'
            });
            return value;
          }
        }
      }
      
      // Check sessionStorage
      for (const key of tokenKeys) {
        const value = sessionStorage.getItem(key);
        if (value) {
          try {
            const parsed = JSON.parse(value);
            if (parsed && parsed.token) {
              console.log(`Found token in sessionStorage.${key}.token`);
              setAuthStatus({
                isLoggedIn: true,
                tokenSource: `sessionStorage.${key}.token`,
                tokenPreview: parsed.token.substring(0, 10) + '...'
              });
              return parsed.token;
            }
          } catch (e) {
            console.log(`Found potential token in sessionStorage.${key}`);
            setAuthStatus({
              isLoggedIn: true,
              tokenSource: `sessionStorage.${key}`,
              tokenPreview: value.substring(0, 10) + '...'
            });
            return value;
          }
        }
      }
      
      // No token found
      console.log("No token found in any storage location");
      setAuthStatus({
        isLoggedIn: false,
        tokenSource: null,
        tokenPreview: null
      });
      return null;
    };
    
    // Check auth on mount and when chatbot opens
    const token = checkAuth();
    
    // Set up event listener for storage changes
    window.addEventListener('storage', checkAuth);
    
    // Check auth status periodically
    const interval = setInterval(checkAuth, 5000);
    
    return () => {
      window.removeEventListener('storage', checkAuth);
      clearInterval(interval);
    };
  }, [isOpen]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const getToken = () => {
    // Try to find token in various storage locations and with various keys
    const tokenKeys = ['token', 'userToken', 'authToken', 'jwt', 'accessToken', 'user'];
    
    // Check localStorage
    for (const key of tokenKeys) {
      const value = localStorage.getItem(key);
      if (value) {
        try {
          // If the value is a JSON object (like a user object with a token property)
          const parsed = JSON.parse(value);
          if (parsed && parsed.token) {
            return parsed.token;
          }
        } catch (e) {
          // Not JSON, treat as direct token
          return value;
        }
      }
    }
    
    // Check sessionStorage
    for (const key of tokenKeys) {
      const value = sessionStorage.getItem(key);
      if (value) {
        try {
          const parsed = JSON.parse(value);
          if (parsed && parsed.token) {
            return parsed.token;
          }
        } catch (e) {
          return value;
        }
      }
    }
    
    return null;
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === "") return;
    
    // Add user message
    const newUserMessage = {
      sender: "user",
      text: inputMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setMessages([...messages, newUserMessage]);
    const userInput = inputMessage;
    setInputMessage(""); // Clear the input
    setIsLoading(true);
    
    try {
      // Get token
      const token = getToken();
      
      // For debugging - log token status
      if (token) {
        console.log(`Using token: ${token.substring(0, 10)}...`);
      } else {
        console.log("No token found, proceeding without authentication");
      }
      
      // Call the backend API
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        body: JSON.stringify({ 
          user_input: userInput,
          token: token || "guest" // Use "guest" as fallback if no token
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        console.error('Error response:', errorData);
        const errorMessage = errorData?.error || `Server error: ${response.status}`;
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      console.log("Backend response:", data);
      
      // Add bot response from the backend
      const newBotMessage = {
        sender: "bot",
        text: data.message || data.error || "I didn't understand that. Could you try again?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setMessages(prevMessages => [...prevMessages, newBotMessage]);
    } catch (error) {
      console.error('Error fetching from backend:', error);
      
      // Add error message
      const errorMessage = {
        sender: "bot",
        text: error.message || "Sorry, I'm having trouble connecting to the server. Please try again later.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className={`desktop-chatbot ${isOpen ? 'open' : ''}`}>
      <div className="desktop-chatbot-header">
        <div className="desktop-chatbot-title">
          StakeFit Assistant
          {authStatus.isLoggedIn && (
            <span className="auth-status-indicator logged-in">●</span>
          )}
          {!authStatus.isLoggedIn && (
            <span className="auth-status-indicator logged-out">●</span>
          )}
        </div>
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
        {isLoading && (
          <div className="desktop-message bot-message">
            <div className="desktop-message-content">
              <p>Thinking...</p>
            </div>
          </div>
        )}
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
          disabled={isLoading}
        />
        <button 
          className="desktop-chatbot-send-btn"
          onClick={handleSendMessage}
          disabled={isLoading}
        >
          <img 
            src="https://cdn.builder.io/api/v1/image/assets/TEMP/235096c15e380490658228ad51c7459a1bec2c30?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
            alt="Send"
          />
        </button>
      </div>
      
      {!authStatus.isLoggedIn && (
        <div className="login-reminder">
          <p>You're using the chatbot in guest mode. Some features may be limited.</p>
        </div>
      )}
      
      {/* Debug info - remove in production */}
      {authStatus.isLoggedIn && (
        <div className="debug-info">
          <p>Token source: {authStatus.tokenSource}</p>
          <p>Token preview: {authStatus.tokenPreview}</p>
        </div>
      )}
    </div>
  );
};

export default DesktopChatbot;
