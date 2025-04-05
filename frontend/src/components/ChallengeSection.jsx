import React, { useEffect, useState } from "react";
import ChallengeCard from "./ChallengeCard";
import "../styles/TabSelector.css";
import { useChallengeContext } from "../context/ChallengeContext";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const ChallengeSection = () => {
  const [activeTab, setActiveTab] = useState('steps');
  const[challenges,setChallenges]=useState([]);
  // const { challenges, setSelectedChallenge } = useChallengeContext();
  const navigate = useNavigate();



  const fetchChallenges = async () => {
    try {
      console.log("hello")
      const response = await axios.get("http://localhost:3000/api/challenges/get");
      console.log(response.data);
      setChallenges(response.data);
    } catch (error) {
      console.error("Error fetching challenges:", error);
    }
  };

  useEffect(() => {
    fetchChallenges();
  }, []);




  // Filter challenges based on type
  const filteredChallenges = challenges.filter(challenge => {
    if (activeTab === 'steps') return challenge.type === 'steps';
    if (activeTab === 'strength') return challenge.type === 'strength';
    if (activeTab === 'combo') return challenge.type === 'combo';
    return true;
  });

  const handleChallengeSelect = (challenge) => {
    setSelectedChallenge(challenge);
    navigate('/details');
  };



  return (
    <div className="w-full max-w-[859px] mx-auto rounded-[19px] p-[50px]">
      <div className="mb-10">
        <h2 className="text-white text-3xl font-medium mb-4">
          Challenge Selection
        </h2>
        <p className="text-[#CDCDCD] text-xl">
          Choose a challenge to stake on ETH
        </p>
      </div>
      <div className="StepsStrengthCombo mb-8">
        <div className="tab-container">
          <div className="tab-nav">
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
      <div className="space-y-12">
      {filteredChallenges.map((challenge) => (
          <ChallengeCard
            key={challenge._id}
            challenge={challenge}
            onClick={() => handleChallengeSelect(challenge)}
          />
        ))}
      </div>
    </div>
  );
};

export default ChallengeSection;