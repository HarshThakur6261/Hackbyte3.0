import { useNavigate } from "react-router-dom";
import React, { useEffect, useState } from "react";
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
import { FaHistory } from "react-icons/fa";
import axios from "axios";
import { jwtDecode } from "jwt-decode";

const Community = () => {
  const navigate = useNavigate();
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const { logout, JwtToken } = useAuth();
  const [userData, setUserData] = useState();
  const [userCommunities, setUserCommunities] = useState([]);
  const [allCommunities, setAllCommunities] = useState([]);
  const [selectedCommunity, setSelectedCommunity] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showJoinModal, setShowJoinModal] = useState(false);
  const [newCommunityName, setNewCommunityName] = useState("");
  const [newCommunityTag, setNewCommunityTag] = useState("");
  const [communitiesRanking, setCommunitiesRanking] = useState([]);

  // Fetch user data
  const getUserData = async () => {
    try {
      const payload = jwtDecode(JwtToken);
      const response = await axios.get(
        `http://localhost:3000/api/users/get/${payload.email}`,
      );
      setUserData(response.data);
    } catch (error) {
      console.log(error);
    }
  };

  // Fetch user communities
  const getUserCommunities = async () => {
    try {
      const response = await axios.get(
        `http://localhost:3000/community/user/${JwtToken}`,
        {
          headers: {
            Authorization: `Bearer ${JwtToken}`,
          },
        }
      );
      setUserCommunities(response.data);
    } catch (error) {
      console.log("Error fetching user communities:", error);
    }
  };

  // Fetch all communities for leaderboard
  const getAllCommunities = async () => {
    try {
      const response = await axios.get("http://localhost:3000/community/all");
      setAllCommunities(response.data); 
      
      // Mock ranking calculation - in real app, this would be based on actual user earnings
      const ranked = response.data.map(community => ({
        ...community,
        avgEarnings: calculateMockAvgEarnings(community)
      })).sort((a, b) => b.avgEarnings - a.avgEarnings)
        .slice(0, 10);  // Top 10
      
      setCommunitiesRanking(ranked);
    } catch (error) {
      console.log("Error fetching all communities:", error);
    }
  };

  // Mock function to calculate average earnings (replace with real calculation)
  const calculateMockAvgEarnings = (community) => {
    // In real implementation, you would calculate based on actual user earnings
    // For now, just generate random values for demonstration
    return parseFloat((Math.random() * 200 + 50).toFixed(2));
  };

  // Join a community
  const handleJoinCommunity = async (communityId) => {
    try {
      await axios.post(
        "http://localhost:3000/api/community/join/:token",
        { communityId },
        {
          headers: {
            Authorization: `Bearer ${JwtToken}`,
          },
        }
      );
      getUserCommunities(); // Refresh user communities
      setShowJoinModal(false);
    } catch (error) {
      console.log("Error joining community:", error);
      alert(error.response?.data?.message || "Failed to join community");
    }
  };

  // Create a community
  const handleCreateCommunity = async () => {
    try {
      await axios.post(
        `http://localhost:3000/community/create/${JwtToken}`,
        { name: newCommunityName, tag: newCommunityTag },
        {
          headers: {
            Authorization: `Bearer ${JwtToken}`,
          },
        }
      );
      setNewCommunityName("");
      setNewCommunityTag("");
      setShowCreateModal(false);
      getUserCommunities(); // Refresh user communities
      getAllCommunities(); // Refresh all communities
    } catch (error) {
      console.log("Error creating community:", error);
      alert(error.response?.data?.message || "Failed to create community");
    }
  };

  // Load data on component mount
  useEffect(() => {
    if (JwtToken) {
      getUserData();
      getUserCommunities();
      getAllCommunities();
    }
  }, [JwtToken]);

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

  // Create Community Modal
  const CreateCommunityModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
      <div className="bg-[#1A0F2B] border-2 border-[#512E8B] rounded-2xl p-6 w-[400px]">
        <h2 className="text-white text-2xl font-medium mb-4">Create Community</h2>
        <div className="mb-4">
          <label className="text-white block mb-2">Community Name</label>
          <input
            type="text"
            value={newCommunityName}
            onChange={(e) => setNewCommunityName(e.target.value)}
            className="w-full bg-[#301F4C] text-white p-2 rounded-lg"
            placeholder="Enter community name"
          />
        </div>
        <div className="mb-6">
          <label className="text-white block mb-2">Community Tag</label>
          <input
            type="text"
            value={newCommunityTag}
            onChange={(e) => setNewCommunityTag(e.target.value)}
            className="w-full bg-[#301F4C] text-white p-2 rounded-lg"
            placeholder="Enter unique tag (no spaces)"
          />
        </div>
        <div className="flex justify-end gap-4">
          <button
            onClick={() => setShowCreateModal(false)}
            className="px-4 py-2 bg-[#3D2A64] text-white rounded-lg hover:bg-[#4a336e]"
          >
            Cancel
          </button>
          <button
            onClick={handleCreateCommunity}
            className="px-4 py-2 bg-[#6b44ae] text-white rounded-lg hover:bg-[#7b54be]"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );

  // Join Community Modal
  const JoinCommunityModal = () => {
    const [availableCommunities, setAvailableCommunities] = useState([]);
    
    useEffect(() => {
      // Filter out communities user has already joined
      const userCommunityIds = userCommunities.map(c => c._id);
      const available = allCommunities.filter(
        c => !userCommunityIds.includes(c._id)
      );
      setAvailableCommunities(available);
    }, [allCommunities, userCommunities]);
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
        <div className="bg-[#1A0F2B] border-2 border-[#512E8B] rounded-2xl p-6 w-[400px]">
          <h2 className="text-white text-2xl font-medium mb-6">Join Community</h2>
          {availableCommunities.length === 0 ? (
            <p className="text-gray-300 mb-6">No communities available to join</p>
          ) : (
            <div className="max-h-[300px] overflow-y-auto mb-6">
              {availableCommunities.map((community) => (
                <div 
                  key={community._id} 
                  className="flex justify-between items-center p-3 mb-2 rounded-lg bg-[#301F4C] hover:bg-[#3D2A64] cursor-pointer"
                  onClick={() => handleJoinCommunity(community._id)}
                >
                  <div>
                    <h3 className="text-white font-medium">{community.name}</h3>
                    <p className="text-gray-300 text-sm">#{community.tag}</p>
                  </div>
                  <button className="px-3 py-1 bg-[#6b44ae] text-white rounded-full text-sm">
                    Join
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex justify-end">
            <button
              onClick={() => setShowJoinModal(false)}
              className="px-4 py-2 bg-[#3D2A64] text-white rounded-lg hover:bg-[#4a336e]"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
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
              <Avatar className="h-[63px] p-3.5 w-[63px] border-4 border-[#512E8B] rounded-full bg-[#413359] cursor-pointer hover:opacity-80 transition-opacity">
                <FaHistory color="white" size={30} />
              </Avatar>
            </button>
          </div>
        </header>

        {/* Community Section */}
        <div className="flex flex-col font-poppins p-6">
          {/* Navigation Buttons */}
          <div className="flex h-[73px] p-3 justify-between items-center rounded-[48px] bg-[#4a336e5c] w-full">
            <div className="flex items-center gap-2">
              <button 
                className="communityjoin px-6 py-2 bg-[#6b44ae] text-white rounded-full hover:bg-[#7b54be]"
                onClick={() => setShowJoinModal(true)}
              >
                Join
              </button>
              <button 
                className="communitycreate px-6 py-2 bg-[#6b44ae] text-white rounded-full hover:bg-[#7b54be]"
                onClick={() => setShowCreateModal(true)}
              >
                Create
              </button>
            </div>
            <select 
              className="w-[180px] bg-[#6b44ae] h-10 pl-2 mr-3 pr-4 py-2 rounded-full border-input text-sm text-white"
              value={selectedCommunity}
              onChange={(e) => setSelectedCommunity(e.target.value)}
            >
              <option value="" disabled>Select Community</option>
              {userCommunities.map((community) => (
                <option key={community._id} value={community._id}>
                  {community.name}
                </option>
              ))}
            </select>  
          </div>

          {/* Main Content */}
          <div className="flex gap-10 mt-8">
            {/* Joined Communities */}
            <div className="flex-1 p-12 rounded-[19px] bg-[#4a336e5c]">
              <h2 className="text-white text-3xl font-medium mb-10">
                {userCommunities.length > 0 ? "Joined Communities" : "No Communities Joined Yet"}
              </h2>
              <div className="flex flex-col gap-6">
                {userCommunities.length > 0 ? (
                  userCommunities.map((community) => (
                    <div
                      key={community._id}
                      className="flex flex-col justify-center p-4 rounded-[11px] border-2 border-[#301F4C] bg-[#1A0F2B] hover:border-[#512E8B] transition-colors cursor-pointer"
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <h3 className="text-white text-2xl font-medium">
                            {community.name}
                          </h3>
                          <p className="text-[#CDCDCD] text-lg font-medium">
                            #{community.tag}
                          </p>
                        </div>
                        <div className="bg-[#301F4C] px-3 py-1 rounded-full">
                          <span className="text-white text-sm">
                            {community.users?.length || 0} Members
                          </span>
                        </div>
                      </div>
                      <div className="mt-2 flex items-center gap-2">
                        <div className="text-sm text-gray-300">
                          Average Earnings: 
                          <span className="text-green-400 font-medium ml-1">
                            {calculateMockAvgEarnings(community).toFixed(2)} ETH
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-10">
                    <p className="text-gray-300 mb-4">You haven't joined any communities yet.</p>
                    <button 
                      onClick={() => setShowJoinModal(true)}
                      className="px-6 py-2 bg-[#6b44ae] text-white rounded-full hover:bg-[#7b54be]"
                    >
                      Join a Community
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Leaderboard */}
            <div className="w-[308px] p-5 rounded-[19px] h-fit bg-[#4a336e5c]">
              <div className="flex flex-col items-center gap-5">
                <h2 className="text-white text-xl font-medium">Leaderboard</h2>
                <img 
                  src="https://cdn.builder.io/api/v1/image/assets/TEMP/5ee0513993b0ba538478b5f1d93cdfa9972dd1ed" 
                  alt="Trophy" 
                  className="rounded-full w-[99px] h-[99px] bg-gradient-to-br from-purple-600 to-red-400" 
                />
                <div className="w-full p-5 rounded-[11px] border-2 border-[#301F4C] bg-[#1A0F2B]">
                  <div className="flex justify-between mb-3">
                    <span className="text-white text-sm font-medium">
                      Rank
                    </span>
                    <span className="text-[#ABABAB] text-sm font-medium">
                      Out of {allCommunities.length}
                    </span>
                  </div>
                  <div className="flex flex-col gap-3">
                    {communitiesRanking.slice(0, 5).map((community, index) => (
                      <div
                        key={community._id}
                        className="flex items-center gap-2.5 pb-3 border-b border-[#7E7E7E]"
                      >
                        <div
                          className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium"
                          style={{ 
                            backgroundColor: 
                              index === 0 ? "#FFD700" : 
                              index === 1 ? "#C0C0C0" : 
                              index === 2 ? "#CD7F32" : "#2A9D90" 
                          }}
                        >
                          {index + 1}
                        </div>
                        <div className="flex-1">
                          <span className="text-white text-sm block">
                            {community.name}
                          </span>
                          <span className="text-green-400 text-xs">
                            {community.avgEarnings} ETH
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modals */}
      {showCreateModal && <CreateCommunityModal />}
      {showJoinModal && <JoinCommunityModal />}

      {/* Chatbot */}
      <div className="chatbot-bubble" onClick={handleToggleChatbot}>
        <img src={chatbot} alt="Chatbot" />
      </div>

      <DesktopChatbot
        isOpen={isChatbotOpen}
        onClose={() => setIsChatbotOpen(false)}
        messages={chatMessages}
      />
    </div>
  );
};

export default Community;