import { useNavigate } from "react-router-dom";
import React, { useEffect, useState } from "react";
import ConnectWallet from "../components/ConnectWallet";
import DesktopChatbot from "../components/DesktopChatbot";
import "../styles/history.css";
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
import {FaHome, FaHistory, FaFilter, FaSort, FaTrophy, FaChartLine } from "react-icons/fa";
import axios from "axios";

const History = () => {
  const navigate = useNavigate();
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const { logout, JwtToken } = useAuth();
  const [historyData, setHistoryData] = useState({
    stats: {
      totalChallenges: 0,
      successRate: "0%",
      ethEarned: 0,
      ethLost: 0
    },
    historyEntries: []
  });
  const [filterType, setFilterType] = useState("all");
  const [sortOrder, setSortOrder] = useState("newest");
  const [isLoading, setIsLoading] = useState(true);

  const fetchUserHistory = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(
        `http://localhost:3000/history/user/${JwtToken}`
      );
      console.log("history", response.data);
      setHistoryData(response.data);
      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching history data:", error);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUserHistory();
  }, []);

  // Handle filtering and sorting
  const handleFilterChange = (e) => {
    setFilterType(e.target.value);
  };

  const handleSortChange = (e) => {
    setSortOrder(e.target.value);
  };

  // Filter and sort the history entries
  const getFilteredAndSortedEntries = () => {
    let filtered = [...historyData.historyEntries];
    
    // Apply filters
    if (filterType !== "all") {
      filtered = filtered.filter(entry => {
        if (filterType === "completed") return entry.isCompleted;
        if (filterType === "incomplete") return !entry.isCompleted;
        return true;
      });
    }
    
    // Apply sorting
    if (sortOrder === "oldest") {
      filtered.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
    } else if (sortOrder === "highest-stake") {
      filtered.sort((a, b) => b.ethStaked - a.ethStaked);
    } else if (sortOrder === "lowest-stake") {
      filtered.sort((a, b) => a.ethStaked - b.ethStaked);
    }
    
    return filtered;
  };

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
  
  const handleNavigateToHome = () => {
    navigate("/");
  };

  const handleToggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  // Calculate progress for each challenge
  const calculateProgress = (exercises) => {
    if (!exercises || exercises.length === 0) return "0/0";
    const total = exercises.length;
    const completed = exercises.filter(ex => ex.isCompleted).length;
    return `${completed}/${total}`;
  };

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toISOString().split('T')[0];
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#2D1B4F] to-[#1A0F2E] text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Header */}
        <header className="flex justify-between items-center py-4 mb-8">
          <div 
            className="desktop-home__logo-container flex items-center gap-2 cursor-pointer"
            onClick={handleNavigateToHome}
          >
            <img
              src="https://cdn.builder.io/api/v1/image/assets/TEMP/69e8365158abd202fc7d010edd0471beda6cd6aa?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
              alt="Logo"
              className="h-10 w-10"
            />
            <div className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-500">StakeFit</div>
          </div>
          <div className="flex gap-4 items-center">
            <div className="overflow-hidden">
              <ConnectWallet />
            </div>

            <button onClick={handleNavigateToHome} className="rounded-full">
              <Avatar className="h-[50px] p-3 w-[50px] border-2 border-purple-500 rounded-full bg-[#413359] cursor-pointer hover:opacity-80 transition-opacity hover:border-pink-500 transform hover:scale-105">
                <FaHome color="white" size={20} />
              </Avatar>
            </button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Avatar className="h-[50px] w-[50px] cursor-pointer hover:opacity-80 transition-opacity border-2 border-purple-500 hover:border-pink-500 transform hover:scale-105">
                  <AvatarImage src="https://github.com/shadcn.png" alt="User" />
                  <AvatarFallback className="bg-[#413359]">US</AvatarFallback>
                </Avatar>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48 bg-[#2D1B4F] border border-purple-500 text-white">
                <DropdownMenuItem
                  className="cursor-pointer hover:bg-[#413359]"
                  onClick={handleLogout}
                >
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex flex-col gap-10">
          {/* Page Header with Controls */}
          <div className="bg-[#2A1A47] rounded-xl p-6 shadow-lg border border-purple-800">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
              <div className="flex items-center gap-3">
                <FaHistory className="text-purple-400 text-2xl" />
                <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-500">
                  Challenge History
                </h1>
              </div>
              <div className="flex gap-4 items-center">
                <div className="relative">
                  <FaFilter className="absolute left-3 top-3 text-purple-400" />
                  <select 
                    className="w-[180px] pl-9 pr-4 py-2 rounded-lg border border-purple-500 bg-[#413359] text-white focus:outline-none focus:ring-2 focus:ring-purple-600 transition-all"
                    value={filterType}
                    onChange={handleFilterChange}
                  >
                    <option value="all">All Challenges</option>
                    <option value="completed">Completed</option>
                    <option value="incomplete">Incomplete</option>
                  </select>
                </div>
                <div className="relative">
                  <FaSort className="absolute left-3 top-3 text-purple-400" />
                  <select 
                    className="w-[180px] pl-9 pr-4 py-2 rounded-lg border border-purple-500 bg-[#413359] text-white focus:outline-none focus:ring-2 focus:ring-purple-600 transition-all"
                    value={sortOrder}
                    onChange={handleSortChange}
                  >
                    <option value="newest">Newest First</option>
                    <option value="oldest">Oldest First</option>
                    <option value="highest-stake">Highest Stake</option>
                    <option value="lowest-stake">Lowest Stake</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-[#3D2963] to-[#2D1B4F] p-4 rounded-lg border border-purple-700 shadow-md hover:shadow-lg transition-all hover:border-purple-500">
                <div className="flex items-center gap-3 mb-2">
                  <FaTrophy className="text-yellow-400" />
                  <h3 className="text-lg font-medium text-gray-200">Total Challenges</h3>
                </div>
                <p className="text-3xl font-bold text-white">{historyData.stats.totalChallenges.toString()}</p>
              </div>
              
              <div className="bg-gradient-to-br from-[#3D2963] to-[#2D1B4F] p-4 rounded-lg border border-purple-700 shadow-md hover:shadow-lg transition-all hover:border-purple-500">
                <div className="flex items-center gap-3 mb-2">
                  <FaChartLine className="text-blue-400" />
                  <h3 className="text-lg font-medium text-gray-200">Success Rate</h3>
                </div>
                <p className="text-3xl font-bold text-white">{historyData.stats.successRate}</p>
              </div>
              
              <div className="bg-gradient-to-br from-[#3D2963] to-[#2D1B4F] p-4 rounded-lg border border-purple-700 shadow-md hover:shadow-lg transition-all hover:border-purple-500">
                <div className="flex items-center gap-3 mb-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-400">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M14.31 8l5.74 9.94M9.69 8h11.48M7.38 12l5.74-9.94M9.69 16L3.95 6.06M14.31 16H2.83M16.62 12l-5.74 9.94"/>
                  </svg>
                  <h3 className="text-lg font-medium text-gray-200"> Earned</h3>
                </div>
                <p className="text-3xl font-bold text-green-400">+ {historyData.stats.ethEarned.toFixed(2)} </p>
              </div>
              
              <div className="bg-gradient-to-br from-[#3D2963] to-[#2D1B4F] p-4 rounded-lg border border-purple-700 shadow-md hover:shadow-lg transition-all hover:border-purple-500">
                <div className="flex items-center gap-3 mb-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-red-400">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M14.31 8l5.74 9.94M9.69 8h11.48M7.38 12l5.74-9.94M9.69 16L3.95 6.06M14.31 16H2.83M16.62 12l-5.74 9.94"/>
                  </svg>
                  <h3 className="text-lg font-medium text-gray-200"> Lost</h3>
                </div>
                <p className="text-3xl font-bold text-red-400">- {historyData.stats.ethLost.toFixed(2)} </p>
              </div>
            </div>
          </div>

          {/* Challenge Cards */}
          <div className="bg-[#2A1A47] rounded-xl p-6 shadow-lg border border-purple-800">
            <h2 className="text-xl font-semibold mb-4 text-gray-200">Your Challenges</h2>
            
            {isLoading ? (
              <div className="flex justify-center items-center h-40">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
              </div>
            ) : getFilteredAndSortedEntries().length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {getFilteredAndSortedEntries().map((entry, index) => (
                  <div key={index} className={`bg-gradient-to-br ${entry.isCompleted ? 'from-[#263B41] to-[#1A272A]' : 'from-[#3D1A1A] to-[#2A0A0A]'} rounded-lg overflow-hidden shadow-lg border ${entry.isCompleted ? 'border-green-700 hover:border-green-500' : 'border-red-700 hover:border-red-500'} hover:shadow-xl transition-all transform hover:scale-[1.02]`}>
                    <div className={`px-4 py-3 ${entry.isCompleted ? 'bg-green-900/20' : 'bg-red-900/20'} border-b ${entry.isCompleted ? 'border-green-700' : 'border-red-700'} flex justify-between items-center`}>
                      <h3 className="font-bold text-lg truncate">{entry.challengeId?.name || "Challenge"}</h3>
                      <span className={`px-2 py-1 rounded-full text-xs ${entry.isCompleted ? 'bg-green-700 text-green-100' : 'bg-red-700 text-red-100'}`}>
                        {entry.isCompleted ? "Completed" : "Not Completed"}
                      </span>
                    </div>
                    <div className="p-4">
                      <div className="flex justify-between mb-2">
                        <span className="text-gray-300">Category:</span>
                        <span className="font-medium">{entry.challengeId?.category || "Fitness"}</span>
                      </div>
                      <div className="flex justify-between mb-2">
                        <span className="text-gray-300">Duration:</span>
                        <span className="font-medium">{entry.challengeId?.duration || "1"} days</span>
                      </div>
                      <div className="flex justify-between mb-2">
                        <span className="text-gray-300">Started on:</span>
                        <span className="font-medium">{formatDate(entry.createdAt)}</span>
                      </div>
                      <div className="flex justify-between mb-4">
                        <span className="text-gray-300">Progress:</span>
                        <span className="font-medium">{calculateProgress(entry.exercises)}</span>
                      </div>
                      
                      {/* Progress bar */}
                      <div className="w-full bg-gray-700 rounded-full h-2.5 mb-4">
                        <div 
                          className={`h-2.5 rounded-full ${entry.isCompleted ? 'bg-green-500' : 'bg-red-500'}`}
                          style={{ width: `${entry.exercises && entry.exercises.length > 0 
                            ? (entry.exercises.filter(ex => ex.isCompleted).length / entry.exercises.length) * 100 
                            : 0}%` 
                          }}
                        ></div>
                      </div>
                      
                      <div className={`flex justify-between items-center p-2 rounded-lg ${entry.isCompleted ? 'bg-green-900/20 border border-green-700' : 'bg-red-900/20 border border-red-700'}`}>
                        <span className="font-medium">{entry.isCompleted ? "Rewards:" : "Stake:"}</span>
                        <span className={`text-lg font-bold ${entry.isCompleted ? 'text-green-400' : 'text-red-400'}`}>
                          {entry.isCompleted ? `+${entry.rewardsEarned.toFixed(2)}` : `${entry.ethStaked.toFixed(2)}`} 
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-[#3D2963] rounded-lg p-8 text-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-purple-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <h3 className="text-xl font-medium mb-2">No challenges found</h3>
                <p className="text-gray-400 mb-4">Start a challenge to see your progress here!</p>
                <button 
                  onClick={() => navigate('/')}
                  className="inline-flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                >
                  <span>Find Challenges</span>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Floating Chatbot Bubble */}
      <div className="fixed right-6 bottom-6 z-50">
        <button 
          onClick={handleToggleChatbot}
          className="bg-gradient-to-r from-purple-600 to-purple-800 w-16 h-16 rounded-full flex items-center justify-center shadow-lg hover:shadow-xl transition-all transform hover:scale-110 border-2 border-purple-400"
        >
          <img src={chatbot} alt="Chatbot" className="w-10 h-10" />
        </button>
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