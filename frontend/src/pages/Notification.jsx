import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const NotificationItem = ({ notification, onAccept, onReject }) => {
  // Determine if this is a challenge notification that needs action buttons
  const isChallengeInvite = notification.type === 'challenge_invite' && !notification.responded;
  
  return (
    <div className="border rounded-lg p-4 mb-3 bg-white shadow-sm hover:shadow-md transition">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-semibold text-lg">{notification.title}</h3>
          <p className="text-gray-600">{notification.description}</p>
          
          {isChallengeInvite && (
            <div className="mt-2">
              <p className="font-medium">Challenge Details:</p>
              <ul className="text-sm text-gray-700 ml-2">
                <li>Stake amount: {notification.data.stakeAmount} ETH</li>
                <li className="truncate max-w-xs">Description: {notification.data.challengeDescription}</li>
              </ul>
            </div>
          )}
          
          <p className="text-xs text-gray-500 mt-2">
            {new Date(notification.createdAt).toLocaleString()}
          </p>
        </div>
        
        {isChallengeInvite && (
          <div className="flex space-x-2">
            <button 
              onClick={() => onAccept(notification)}
              className="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded-md text-sm"
            >
              Accept
            </button>
            <button 
              onClick={() => onReject(notification)}
              className="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded-md text-sm"
            >
              Reject
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

const NotificationsPanel = () => {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Fetch user notifications
  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        const token = localStorage.getItem('jwtToken');
        const response = await axios.get('http://localhost:3000/notifications/recent', {
          headers: { Authorization: `Bearer ${token}` }
        });
        console.log(response);
        setNotifications(response.data);
      } catch (error) {
        console.error('Error fetching notifications:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchNotifications();
    
    // Optional: Set up polling to check for new notifications
    const interval = setInterval(fetchNotifications, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Handle accepting a challenge
  const handleAccept = async (notification) => {
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
      
      // Update the notification in the UI
      setNotifications(prevNotifications =>
        prevNotifications.map(n =>
          n._id === notification._id ? { ...n, responded: true } : n
        )
      );
      
      // Optionally navigate to the challenge page
      navigate(`/community-challenge/${notification.data.challengeId}`);
    } catch (error) {
      console.error('Error accepting challenge:', error);
      alert('Failed to accept challenge. Please try again.');
    }
  };

  // Handle rejecting a challenge
  const handleReject = async (notification) => {
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
      
      // Update the notification in the UI
      setNotifications(prevNotifications =>
        prevNotifications.map(n =>
          n._id === notification._id ? { ...n, responded: true } : n
        )
      );
    } catch (error) {
      console.error('Error rejecting challenge:', error);
      alert('Failed to reject challenge. Please try again.');
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h2 className="text-2xl font-bold mb-6">Notifications</h2>
      
      {loading ? (
        <div className="text-center py-4">Loading notifications...</div>
      ) : notifications.length === 0 ? (
        <div className="text-center py-4 text-gray-500">No notifications</div>
      ) : (
        notifications.map(notification => (
          <NotificationItem
            key={notification._id}
            notification={notification}
            onAccept={handleAccept}
            onReject={handleReject}
          />
        ))
      )}
    </div>
  );
};

export default NotificationsPanel;