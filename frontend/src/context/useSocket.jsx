// In a context provider or app initialization
import React, { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const SocketContext = createContext();

export const SocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    // Initialize socket connection
    const socketInstance = io('http://localhost:3000');
    setSocket(socketInstance);

    // Clean up on unmount
    return () => {
      socketInstance.disconnect();
    };
  }, []);

  // Authenticate socket when token is available
  useEffect(() => {
    if (socket) {
      const token = localStorage.getItem('token');
      if (token) {
        socket.emit('authenticate', token);
      }
    }
  }, [socket]);

  // Listen for new notifications
  useEffect(() => {
    if (socket) {
      socket.on('new_notification', (notification) => {
        // Add new notification to state
        setNotifications(prev => [notification, ...prev]);
        
        // Show browser notification if permission granted
        if (Notification.permission === 'granted') {
          new Notification(notification.title, {
            body: notification.description
          });
        }
      });
    }

    return () => {
      if (socket) {
        socket.off('new_notification');
      }
    };
  }, [socket]);

  return (
    <SocketContext.Provider value={{ socket, notifications, setNotifications }}>
      {children}
    </SocketContext.Provider>
  );
};

export const useSocket = () => useContext(SocketContext);