import React, { useEffect, useState } from "react";
import ChallengeCard from "./ChallengeCard";
import "../styles/TabSelector.css";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const ChallengeSection = ({ userData }) => {
  console.log(userData)
  const [activeTab, setActiveTab] = useState('all');
  const [challenges, setChallenges] = useState([]);
  const [recommendationData, setRecommendationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Fetch regular challenges
  const fetchChallenges = async () => {
    try {
      const response = await axios.get("http://localhost:3000/api/challenges/get");
      console.log(response.data);
      setChallenges(response.data);
    } catch (error) {
      console.error("Error fetching challenges:", error);
    }
  };

  // Fetch personalized recommendations
  const fetchRecommendations = async () => {
    if (!userData) return;
    
    setLoading(true);
    try {
      const response = await axios.post(
        "http://localhost:3000/flask/recommendations", 
        {
          height: userData.height,
          weight: userData.weight,
          age: userData.age,
          gender: userData.gender,
          // goals: userData.goals || []
        },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      if (error.response) {
        console.error("Response data:", error.response.data);
        console.error("Response status:", error.response.status);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChallenges();
    if (userData) {
      fetchRecommendations();
    }
  }, []);

  // Combine all challenges with AI recommendations marked
  const getAllChallenges = () => {
    const regularChallenges = challenges.map(c => ({
      ...c,
      isRecommended: false,
      rawData: null
    }));

    if (!recommendationData) return regularChallenges;

    const aiChallenges = recommendationData.recommendations.map(rec => ({
      _id: `rec-${rec.ChallengeName.toLowerCase().replace(/\s+/g, '-')}`,
      name: rec.ChallengeName,
      type: rec.type === 'cardio' ? 'steps' : 
            rec.type === 'strength' ? 'strength' : 'combo',
      description: rec.desc,
      difficulty: rec.Intensity === 'high' ? 'Hard' : 
                 rec.Intensity === 'medium' ? 'Medium' : 'Easy',
      stepGoal: rec.type === 'cardio' ? 10000 : 5000,
      exercises: rec.Exercises,
      minStake: 0.01,
      maxStake: 0.1,
      rewardMultiplier: rec.Intensity === 'high' ? 2.5 : 
                      rec.Intensity === 'medium' ? 2.0 : 1.5,
      isRecommended: true,
      rawData: rec
    }));

    return [...aiChallenges, ...regularChallenges];
  };

  // Filter challenges based on active tab
  const getFilteredChallenges = () => {
    const allChallenges = getAllChallenges();
    if (activeTab === 'all') return allChallenges;
    return allChallenges.filter(c => c.type === activeTab);
  };

  const handleChallengeSelect = (challenge) => {
    navigate('/details', { state: { challenge } });
  };

  return (
    <div className="w-full max-w-[859px] mx-auto rounded-[19px] p-[50px]">
      <div className="mb-10">
        <h2 className="text-white text-3xl font-medium mb-4">
          {userData ? "Recommended For You" : "All Challenges"}
        </h2>
        <p className="text-[#CDCDCD] text-xl">
          {userData ? "Based on your profile" : "Choose a challenge to stake on ETH"}
        </p>
      </div>

      <div className="StepsStrengthCombo mb-8">
        <div className="tab-container">
          <div className="tab-nav">
            {/* <button
              className={`tab-button ${activeTab === 'all' ? 'active' : ''}`}
              onClick={() => setActiveTab('all')}
            >
              All
            </button> */}
            <button
              className={`tab-button ${activeTab === 'steps' ? 'active' : ''}`}
              onClick={() => setActiveTab('steps')}
            >
              Steps
            </button>
            <button
              className={`tab-button ${activeTab === 'strength' ? 'active' : ''}`}
              onClick={() => setActiveTab('strength')}
            >
              Strength
            </button>
            <button
              className={`tab-button ${activeTab === 'combo' ? 'active' : ''}`}
              onClick={() => setActiveTab('combo')}
            >
              Combo
            </button>
          </div>
        </div>
      </div>

      {loading && (
        <div className="text-white text-center py-8">Loading recommendations...</div>
      )}

      <div className="space-y-12">
        {getFilteredChallenges().map((challenge) => (
          <ChallengeCard
            key={challenge._id}
            challenge={challenge}
            isRecommended={challenge.isRecommended}
            onClick={() => handleChallengeSelect(challenge)}
          />
        ))}
      </div>
    </div>
  );
};

export default ChallengeSection;