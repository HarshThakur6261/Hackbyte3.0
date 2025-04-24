import React, { useState, useEffect } from "react";
import axios from "axios";
import { useParams, useNavigate, useLocation  } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const CommunityChallenge = () => {
  const location = useLocation(); 
  const communityId = location.state?.communityId;
  // Get community ID from URL params
  const navigate = useNavigate();
  
  const [showModal, setShowModal] = useState(false);
  const [challengeName, setChallengeName] = useState("");
  const [challengeType, setChallengeType] = useState("combo");
  const [stepGoal, setStepGoal] = useState("");
  const [exercises, setExercises] = useState([
    { 
      name: "", 
      reps: "", 
      isVideoRequired: false, 
      videoUrl: ""
    },
  ]);
  const [minStake, setMinStake] = useState("0.00001");
  const [maxStake, setMaxStake] = useState("21");
  const [rewardMultiplier, setRewardMultiplier] = useState("1.2");
  const [difficulty, setDifficulty] = useState("medium");
  const [tags, setTags] = useState(["fitness", "daily"]);
  const [todayChallenge, setTodayChallenge] = useState(null);
  const [loading, setLoading] = useState(true);
  const [ethStaked, setEthStaked] = useState("");
  const { logout, JwtToken } = useAuth();
  
  // New state variables
  const [communityInfo, setCommunityInfo] = useState(null);
  const [communityMembers, setCommunityMembers] = useState([]);
  const [isUserLeader, setIsUserLeader] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);

  // Fetch community data, challenge, and users on component mount
  useEffect(() => {
    console.log(communityId);
    const fetchAllData = async () => {
      try {
        setLoading(true);
        await Promise.all([
          fetchCommunityInfo(),
          fetchTodayChallenge(),
          fetchCurrentUser(),
        ]);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };
    
    if (communityId) {
      console.log("enter");
      fetchAllData();
    }
  }, [communityId]);

  // Fetch additional data once we have the user and community info
  useEffect(() => {
    if (communityInfo && currentUser) {
      fetchCommunityMembers();
      checkIfUserIsLeader();
      fetchLeaderboard();
    }
  }, [communityInfo, currentUser]);

  const fetchCurrentUser = async () => {
    try {
     
      // Replace with your actual API endpoint
      const response = await axios.get(`http://localhost:3000/community/users/${JwtToken}`);
      console.log(response.data);
      setCurrentUser(response.data);
      return response.data.user;
    } catch (error) {
      console.error("Error fetching current user:", error);
      return null;
    }
  };

  const fetchCommunityInfo = async () => {
    try {
      console.log("enter");
      const response = await axios.get(`http://localhost:3000/community/${communityId}`);
      console.log(response.data);
      setCommunityInfo(response.data);
      return response.data.community;
    } catch (error) {
      console.error("Error fetching community info:", error);
      return null;
    }
  };

  const fetchCommunityMembers = async () => {
    try {
      const token = localStorage.getItem("JwtToken");
      const response = await axios.get(
        `http://localhost:3000/community/${communityId}/members`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
  
      console.log("Members data:", response.data);
  
      // Process and sort members by points (descending)
      const processedMembers = response.data.map(member => ({
        ...member,
        points: member.points || 0  // Ensure points has a default value
      }));
  
      const sortedMembers = [...processedMembers].sort((a, b) => b.points - a.points);
  
      setCommunityMembers(processedMembers);
      setLeaderboard(sortedMembers); // Also update leaderboard with sorted data
  
    } catch (error) {
      console.error("Error fetching community members:", error);
      // Optional: Add error state handling
      setCommunityMembers([]);
      setLeaderboard([]);
    }
  };
  
  const fetchLeaderboard = async () => {
    try {
      const token = localStorage.getItem("JwtToken");
      const response = await axios.get(
        `http://localhost:3000/communityChallenge/community/${communityId}/leaderboard`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
  
      // Process leaderboard data - ensure points exist and sort
      const processedLeaderboard = (response.data.leaderboard || []).map(item => ({
        ...item,
        points: item.points || 0
      })).sort((a, b) => b.points - a.points);
  
      setLeaderboard(processedLeaderboard);
  
    } catch (error) {
      console.error("Error fetching leaderboard:", error);
      // Optional: Add error state handling
      setLeaderboard([]);
    }
  };

  const checkIfUserIsLeader = () => {
    if (currentUser && communityInfo) {
      setIsUserLeader(currentUser._id === communityInfo.leaderId);
    }
  };

  const fetchTodayChallenge = async () => {
    try {
      const response = await axios.get(`http://localhost:3000/todaychallenge/${communityId}`);
      console.log("Hyyyyyyyy")
      console.log(response.data)
      if (response.data) {
        setTodayChallenge(response.data);
      } else {
        setTodayChallenge(null);
        setShowModal(true)
      }
    } catch (error) {
      console.error("Error fetching today's challenge:", error);
      setTodayChallenge(null);
    }
  };

  const addExercise = () => {
    setExercises([
      ...exercises,
      { name: "", reps: "", isVideoRequired: false, videoUrl: "" },
    ]);
  };

  const updateExercise = (index, field, value) => {
    const updated = [...exercises];
    updated[index][field] = value;
    setExercises(updated);
  };

  const toggleVideoRequired = (index) => {
    const updated = [...exercises];
    updated[index].isVideoRequired = !updated[index].isVideoRequired;
    setExercises(updated);
  };

  const handleTagChange = (e) => {
    const tagValue = e.target.value;
    setTags(tagValue.split(',').map(tag => tag.trim()));
  };

  const handleCreateChallenge = async () => {
    try {
      // Create the challenge object matching the schema
      const newChallenge = {
        communityId: communityId,
        name: challengeName,
        type: challengeType,
        stepGoal: parseInt(stepGoal) || 0,
        exercises: exercises.filter(ex => ex.name).map(exercise => ({
          name: exercise.name,
          reps: parseInt(exercise.reps) || 0,
          isVideoRequired: exercise.isVideoRequired,
          videoUrl: exercise.videoUrl || ""
        })),
        ethStaked: parseFloat(ethStaked) || 0,
        minStake: minStake.toString(),
        maxStake: maxStake.toString(),
        rewardMultiplier: parseFloat(rewardMultiplier),
        difficulty: difficulty,
        tags: tags
      };

      console.log("Submitting challenge:", newChallenge);

      const response = await axios.post(
        "http://localhost:3000/communityChallenge/create", 
        newChallenge
      );

      if (response.status === 200) {
        alert("Challenge created successfully!");
        setShowModal(false);
        // Reset form
        resetForm();
        // Fetch the newly created challenge
        fetchTodayChallenge();
      } else {
        alert("Failed to create challenge.");
      }
    } catch (error) {
      console.error("Error creating challenge:", error);
      alert("An error occurred while creating the challenge.");
    }
  };

  const handletakeonchallange = async (challangeid) => {
   
      try {
      
  
          const stakeData = {
              challengeId: challangeid,
              ethStaked: 5,
              rewardsEarned: 5,
              isCompleted: false
          };
  
          const response = await axios.post(
             `http://localhost:3000/ActiveChallenge/create/${localStorage.getItem('JwtToken')}`,
              stakeData
          );
  
          if (response.status === 201) {
           
              navigate("/");
          } else {
              throw new Error("Failed to create challenge");
          }
      } catch (error) {
          console.error("Staking failed:", error);
          
      } finally {
          setLoading(false);
      }
  };
  

  const resetForm = () => {
    setChallengeName("");
    setChallengeType("combo");
    setStepGoal("");
    setExercises([{ name: "", reps: "", isVideoRequired: false, videoUrl: "" }]);
    setMinStake("0.00001");
    setMaxStake("21");
    setRewardMultiplier("1.2");
    setDifficulty("medium");
    setTags(["fitness", "daily"]);
    setEthStaked("");
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center bg-opacity-90">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-t-purple-500 border-gray-700"></div>
          <p className="mt-4 text-lg text-purple-300">Loading community data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-navy-900 p-6 relative overflow-hidden" style={{ backgroundColor: '#0a1929' }}>
      {/* Glowing corners */}
      <div className="absolute top-0 left-0 w-64 h-64 rounded-full bg-purple-600 opacity-20 blur-3xl pointer-events-none"></div>
      <div className="absolute bottom-0 right-0 w-96 h-96 rounded-full bg-teal-400 opacity-10 blur-3xl pointer-events-none"></div>
      
      {/* Header with Community Name */}
      <div className="bg-gradient-to-r from-purple-900 to-indigo-900 text-white p-6 rounded-xl mb-8 shadow-lg border border-purple-500 border-opacity-20">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-white">{communityInfo?.name || "Community"}</h1>
            <p className="text-purple-200 mt-1">{communityInfo?.description || ""}</p>
          </div>
          <button
            onClick={() => navigate(-1)}
            className="bg-white bg-opacity-10 hover:bg-opacity-20 text-white px-4 py-2 rounded-lg transition border border-white border-opacity-10"
          >
            Back
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Current Challenge Section */}
        <div className="lg:col-span-2">
          <div className="bg-gray-900 bg-opacity-80 rounded-xl shadow-md p-6 mb-8 border border-purple-500 border-opacity-10">
            <h2 className="text-2xl font-semibold mb-4 text-white">Today's Challenge</h2>
            
            {todayChallenge ? (
              <div className="bg-gradient-to-r from-purple-900 to-indigo-900 p-5 rounded-lg border border-purple-500 border-opacity-20">
                <h3 className="text-xl font-semibold text-purple-200 mb-2">{todayChallenge.name}</h3>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-black bg-opacity-30 p-3 rounded-lg border border-purple-500 border-opacity-10">
                    <p className="text-sm text-purple-300">Type</p>
                    <p className="font-medium capitalize text-white">{todayChallenge.type}</p>
                  </div>
                  <div className="bg-black bg-opacity-30 p-3 rounded-lg border border-purple-500 border-opacity-10">
                    <p className="text-sm text-purple-300">Difficulty</p>
                    <p className="font-medium capitalize text-white">{todayChallenge.difficulty}</p>
                  </div>
                </div>
                
                {todayChallenge.stepGoal > 0 && (
                  <div className="bg-black bg-opacity-30 p-3 rounded-lg mb-4 border border-purple-500 border-opacity-10">
                    <p className="text-sm text-purple-300">Step Goal</p>
                    <p className="font-medium text-white">{todayChallenge.stepGoal.toLocaleString()} steps</p>
                  </div>
                )}
                
                {todayChallenge.exercises && todayChallenge.exercises.length > 0 && (
                  <div className="bg-black bg-opacity-30 p-3 rounded-lg mb-4 border border-purple-500 border-opacity-10">
                    <p className="text-sm text-purple-300 mb-2">Exercises</p>
                    <ul className="space-y-2">
                      {todayChallenge.exercises.map((ex, i) => (
                        <li key={i} className="flex justify-between text-white">
                          <span>{ex.name}</span>
                          <span className="font-medium">{ex.reps} reps</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-black bg-opacity-30 p-3 rounded-lg border border-purple-500 border-opacity-10">
                    <p className="text-sm text-purple-300">Min Stake</p>
                    <p className="font-medium text-white">{todayChallenge.minStake} ETH</p>
                  </div>
                  <div className="bg-black bg-opacity-30 p-3 rounded-lg border border-purple-500 border-opacity-10">
                    <p className="text-sm text-purple-300">Max Stake</p>
                    <p className="font-medium text-white">{todayChallenge.maxStake} ETH</p>
                  </div>
                </div>
                
                <button 
                  onClick={() => handletakeonchallange(todayChallenge._id)}
                  className="mt-6 w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white font-medium py-3 px-4 rounded-lg transition shadow-lg border border-purple-500 border-opacity-30"
                >
                  Take On Challenge
                </button>
              </div>
            ) : (
              <div className="bg-gray-800 p-5 rounded-lg text-center border border-purple-500 border-opacity-10">
                <p className="text-gray-300 mb-4">No challenge exists for today.</p>
                
                {isUserLeader && (
                  <button
                    onClick={() => setShowModal(true)}
                    className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white px-6 py-3 rounded-lg transition shadow-md border border-purple-500 border-opacity-30"
                  >
                    + Create Today's Challenge
                  </button>
                )}
              </div>
            )}
          </div>


          <button onClick={()=>setShowModal(!showModal)} className="text-white text-xl mb-1.5">create Challenge</button>
          
          {/* Community Members Section */}
          <div className="bg-gray-900 bg-opacity-80 rounded-xl shadow-md p-6 border border-purple-500 border-opacity-10">
            <h2 className="text-2xl font-semibold mb-4 text-white">Community Members</h2>
            
            {communityMembers.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {communityMembers.map((member, index) => (
                  <div key={index} className="bg-gray-800 bg-opacity-70 p-4 rounded-lg flex items-center border border-purple-500 border-opacity-10">
                    <div className="bg-gradient-to-r from-purple-600 to-indigo-600 w-10 h-10 rounded-full flex items-center justify-center text-white font-medium mr-3 shadow-lg">
                      {member.userId?.name ? member.userId.name.charAt(0).toUpperCase() : "U"}
                    </div>
                    <div>
                      <p className="font-medium text-white">{member.userId?.name || "User"}</p>
                      <p className="text-sm text-gray-400 truncate">{member.userId?.walletAddress || "No Wallet"}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-400 text-center py-4">No members found in this community.</p>
            )}
          </div>
        </div>
        
        {/* Leaderboard Section */}
        <div className="bg-gray-900 bg-opacity-80 rounded-xl shadow-md p-6 h-fit border border-purple-500 border-opacity-10">
          <h2 className="text-2xl font-semibold mb-4 text-white flex items-center">
            <span className="mr-2">🏆</span> Leaderboard
          </h2>
          
          {leaderboard.length > 0 ? (
            <div className="space-y-4">
              {leaderboard.map((user, index) => (
                <div 
                  key={index} 
                  className={`p-4 rounded-lg flex items-center justify-between border ${
                    index === 0 ? 'bg-gradient-to-r from-yellow-900 to-yellow-800 border-yellow-600' :
                    index === 1 ? 'bg-gradient-to-r from-gray-800 to-gray-700 border-gray-500' :
                    index === 2 ? 'bg-gradient-to-r from-amber-900 to-amber-800 border-amber-700' :
                    'bg-gray-800 border-purple-500 border-opacity-10'
                  } border-opacity-30`}
                >
                  <div className="flex items-center">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold mr-3 ${
                      index === 0 ? 'bg-yellow-500' :
                      index === 1 ? 'bg-gray-400' :
                      index === 2 ? 'bg-amber-600' :
                      'bg-purple-600'
                    }`}>
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-white">{user.userId?.name || "User"}</p>
                      <p className="text-xs text-gray-400 truncate">{user.userId?.walletAddress || "-"}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`font-bold text-lg ${
                      index === 0 ? 'text-yellow-400' :
                      index === 1 ? 'text-gray-300' :
                      index === 2 ? 'text-amber-400' :
                      'text-purple-300'
                    }`}>{user.points || 0}</p>
                    <p className="text-xs text-gray-400">points</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-gray-800 p-5 rounded-lg text-center border border-purple-500 border-opacity-10">
              <p className="text-gray-400">No leaderboard data available.</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Challenge Modal */}
      {showModal && (
  <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-90">
    <div className="bg-[#1A0F2B] rounded-2xl p-8 w-full max-w-2xl shadow-xl overflow-y-auto max-h-[90vh] border-2 border-[#512E8B]">
      <h2 className="text-2xl font-bold mb-6 text-center text-white">
        Create Today's Challenge
      </h2>

      <div className="space-y-6">
        {/* Challenge Name */}
        <div>
          <label className="block text-sm font-medium mb-2 text-purple-300">
            Challenge Name*
          </label>
          <input
            type="text"
            value={challengeName}
            onChange={(e) => setChallengeName(e.target.value)}
            className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            required
          />
        </div>

        {/* Challenge Type */}
        <div>
          <label className="block text-sm font-medium mb-2 text-purple-300">
            Challenge Type*
          </label>
          <select
            value={challengeType}
            onChange={(e) => setChallengeType(e.target.value)}
            className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="steps">Steps Only</option>
            <option value="exercise">Exercise Only</option>
            <option value="combo">Combo (Steps + Exercise)</option>
          </select>
        </div>

        {/* Step Goal (conditionally shown) */}
        {(challengeType === "steps" || challengeType === "combo") && (
          <div>
            <label className="block text-sm font-medium mb-2 text-purple-300">
              Step Goal
            </label>
            <input
              type="number"
              value={stepGoal}
              onChange={(e) => setStepGoal(e.target.value)}
              placeholder="e.g., 6000"
              className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
        )}

        {/* Exercises Section */}
        <div className="space-y-4">
          <label className="block text-sm font-medium text-purple-300">
            Exercises*
          </label>
          
          {exercises.map((exercise, index) => (
            <div key={index} className="p-4 bg-[#2B1748] rounded-lg border border-[#3A225D] space-y-3">
              <input
                type="text"
                value={exercise.name}
                onChange={(e) => updateExercise(index, "name", e.target.value)}
                placeholder="Exercise Name*"
                className="w-full px-4 py-2 bg-[#3A225D] text-white rounded-lg border border-[#4A2D7A]"
                required
              />
              <input
                type="number"
                value={exercise.reps}
                onChange={(e) => updateExercise(index, "reps", e.target.value)}
                placeholder="Required Reps*"
                className="w-full px-4 py-2 bg-[#3A225D] text-white rounded-lg border border-[#4A2D7A]"
                required
              />
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id={`video-required-${index}`}
                  checked={exercise.isVideoRequired}
                  onChange={() => toggleVideoRequired(index)}
                  className="h-4 w-4 rounded border-[#3A225D] bg-[#2B1748] text-purple-500 focus:ring-purple-500"
                />
                <label htmlFor={`video-required-${index}`} className="ml-2 text-sm text-gray-300">
                  Video Required for Verification
                </label>
              </div>
            </div>
          ))}

          <button
            type="button"
            onClick={addExercise}
            className="w-full py-2 bg-[#3A225D] text-purple-300 hover:bg-[#4A2D7A] hover:text-white rounded-lg border border-dashed border-[#512E8B] flex items-center justify-center gap-2"
          >
            <span className="text-lg">+</span> Add Exercise
          </button>
        </div>

        {/* Stake Amount */}
        <div>
          <label className="block text-sm font-medium mb-2 text-purple-300">
            ETH Stake Amount*
          </label>
          <input
            type="text"
            value={ethStaked}
            onChange={(e) => setEthStaked(e.target.value.replace(/[^0-9.]/g, ''))}
            placeholder="Amount to stake (ETH)"
            className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            pattern="^\d*\.?\d*$"
            required
          />
        </div>

        {/* Stake Range */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2 text-purple-300">
              Min Stake (ETH)*
            </label>
            <input
              type="text"
              value={minStake}
              onChange={(e) => setMinStake(e.target.value.replace(/[^0-9.]/g, ''))}
              placeholder="0.00001"
              className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              pattern="^\d*\.?\d*$"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2 text-purple-300">
              Max Stake (ETH)*
            </label>
            <input
              type="text"
              value={maxStake}
              onChange={(e) => setMaxStake(e.target.value.replace(/[^0-9.]/g, ''))}
              placeholder="21"
              className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              pattern="^\d*\.?\d*$"
              required
            />
          </div>
        </div>

        {/* Reward Multiplier */}
        <div>
          <label className="block text-sm font-medium mb-2 text-purple-300">
            Reward Multiplier*
          </label>
          <input
            type="text"
            value={rewardMultiplier}
            onChange={(e) => setRewardMultiplier(e.target.value.replace(/[^0-9.]/g, ''))}
            placeholder="1.2"
            className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            pattern="^\d*\.?\d*$"
            required
          />
        </div>

        {/* Difficulty */}
        <div>
          <label className="block text-sm font-medium mb-2 text-purple-300">
            Difficulty
          </label>
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>

        {/* Tags */}
        <div>
          <label className="block text-sm font-medium mb-2 text-purple-300">
            Tags (comma-separated)
          </label>
          <input
            type="text"
            value={tags.join(', ')}
            onChange={handleTagChange}
            placeholder="fitness, daily"
            className="w-full px-4 py-3 bg-[#2B1748] text-white rounded-lg border border-[#3A225D] focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        {/* Form Actions */}
        <div className="flex justify-end gap-4 pt-4">
          <button
            type="button"
            onClick={() => setShowModal(true)}
            className="px-6 py-2 bg-[#3A225D] text-gray-300 hover:bg-[#4A2D7A] hover:text-white rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleCreateChallenge}
            disabled={!challengeName || (challengeType !== "steps" && exercises.every(ex => !ex.name))}
            className={`px-6 py-2 rounded-lg transition-colors ${
              !challengeName || (challengeType !== "steps" && exercises.every(ex => !ex.name))
                ? "bg-gray-700 text-gray-500 cursor-not-allowed"
                : "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white"
            }`}
          >
            Create Challenge
          </button>
        </div>
      </div>
    </div>
  </div>
)}
    </div>
  );
};

export default CommunityChallenge;