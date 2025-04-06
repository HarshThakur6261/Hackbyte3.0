import React, { useState, useEffect, useRef } from 'react';
import { useSocket } from '../context/useSocket';
import axios from 'axios';
import { Link } from 'react-router-dom';

const NotificationBell = () => {
  const [unreadCount, setUnreadCount] = useState(0);
  const [showDropdown, setShowDropdown] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(false);
  const { notifications: socketNotifications } = useSocket();
  const dropdownRef = useRef(null);


  // Request browser notification permission
  useEffect(() => {
    if (Notification.permission !== 'granted' && Notification.permission !== 'denied') {
      Notification.requestPermission();
    }
  }, []);

  // Update notifications when new ones arrive via socket
  useEffect(() => {
    if (socketNotifications.length > 0) {
      // Update notification count
      setUnreadCount(prev => prev + 1);
      
      // Update displayed notifications if dropdown is open
      if (showDropdown) {
        setNotifications(prev => {
          // Add new notifications at the top
          const newNotifications = [...socketNotifications, ...prev];
          // Limit to most recent 10
          return newNotifications.slice(0, 10);
        });
      }
    }
  }, [socketNotifications, showDropdown]);

  const fetchUnreadCount = async () => {
    try {
      const token = localStorage.getItem('jwtToken');
      const response = await axios.get('http://localhost:3000/notifications/unread/count', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUnreadCount(response.data.count);
    } catch (error) {
      console.error('Error fetching notification count:', error);
    }
  };

  // Fetch recent notifications for the dropdown
  const fetchRecentNotifications = async () => {
    if (showDropdown) {
      setLoading(true);
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get('http://localhost:3000/notifications/recent', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setNotifications(response.data);
      } catch (error) {
        console.error('Error fetching notifications:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  // Toggle dropdown visibility
  const toggleDropdown = () => {
    setShowDropdown(!showDropdown);
  };

  // Handle quick accept action from dropdown
  const handleQuickAccept = async (notification) => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(
        'http://localhost:3000/notifications/challenge/accept',
        {
          notificationId: notification._id,
          challengeId: notification.data.challengeId
        },
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      
      // Update notifications in the dropdown
      setNotifications(prevNotifications =>
        prevNotifications.map(n =>
          n._id === notification._id ? { ...n, responded: true } : n
        )
      );
      
      // Update the unread count
      fetchUnreadCount();
    } catch (error) {
      console.error('Error accepting challenge:', error);
    }
  };

  // Handle quick reject action from dropdown
  const handleQuickReject = async (notification) => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(
        'http://localhost:3000/notifications/challenge/reject',
        {
          notificationId: notification._id
        },
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      
      // Update notifications in the dropdown
      setNotifications(prevNotifications =>
        prevNotifications.map(n =>
          n._id === notification._id ? { ...n, responded: true } : n
        )
      );
      
      // Update the unread count
      fetchUnreadCount();
    } catch (error) {
      console.error('Error rejecting challenge:', error);
    }
  };

  // Mark all notifications as read
  const markAllAsRead = async () => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(
        'http://localhost:3000/notifications/mark-all-read',
        {},
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
      
      // Update the unread count
      fetchUnreadCount();
    } catch (error) {
      console.error('Error marking notifications as read:', error);
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Fetch notification count on mount and set up polling
  useEffect(() => {
    fetchUnreadCount();
    const interval = setInterval(fetchUnreadCount, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, []);

  // Fetch notifications when dropdown is opened
  useEffect(() => {
    fetchRecentNotifications();
  }, [showDropdown]);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        className="relative p-2 text-gray-600 hover:text-gray-800 focus:outline-none"
        onClick={toggleDropdown}
      >
        {/* Bell Icon */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
          />
        </svg>
        
        {/* Notification Badge */}
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-red-100 transform translate-x-1/2 -translate-y-1/2 bg-red-600 rounded-full">
            {unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown Menu */}
      {showDropdown && (
        <div className="absolute right-0 mt-2 w-80 bg-white rounded-md shadow-lg z-20">
          <div className="py-2 px-4 border-b flex justify-between items-center">
            <h3 className="text-sm font-semibold">Notifications</h3>
            {unreadCount > 0 && (
              <button 
                onClick={markAllAsRead}
                className="text-xs text-blue-600 hover:text-blue-800"
              >
                Mark all as read
              </button>
            )}
          </div>
          <div className="max-h-80 overflow-y-auto">
            {loading ? (
              <div className="py-4 text-center text-sm text-gray-500">Loading...</div>
            ) : notifications.length === 0 ? (
              <div className="py-4 text-center text-sm text-gray-500">No new notifications</div>
            ) : (
              <div>
                {notifications.map(notification => (
                  <div key={notification._id} className={`py-3 px-4 border-b ${!notification.read ? 'bg-blue-50' : ''}`}>
                    <div className="flex justify-between">
                      <p className="text-sm font-medium">{notification.title}</p>
                      <span className="text-xs text-gray-500">
                        {new Date(notification.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mt-1 mb-2">{notification.description}</p>
                    
                    {notification.type === 'challenge_invite' && !notification.responded && (
                      <div className="flex space-x-2 mt-1">
                        <button 
                          onClick={() => handleQuickAccept(notification)}
                          className="bg-green-500 hover:bg-green-600 text-white px-2 py-1 rounded text-xs"
                        >
                          Accept
                        </button>
                        <button 
                          onClick={() => handleQuickReject(notification)}
                          className="bg-red-500 hover:bg-red-600 text-white px-2 py-1 rounded text-xs"
                        >
                          Reject
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="py-2 px-4 text-center border-t">
            <Link 
              to="/notifications"
              className="text-sm text-blue-600 hover:text-blue-800"
              onClick={() => setShowDropdown(false)}
            >
              View all notifications
            </Link>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotificationBell;

