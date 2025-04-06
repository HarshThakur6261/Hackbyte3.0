import React from "react";
import { ArrowRight } from "lucide-react";
import "../styles/DesktopHome.css";
import { useNavigate } from "react-router-dom";
import { useChallengeContext } from "../context/ChallengeContext";

const ChallengeCard = ({ challenge, onClick }) => {
  const navigate = useNavigate();
  // const { setSelectedChallenge } = useChallengeContext();

  const getDifficultyStyles = (difficulty) => {
    const baseStyles = "text-white text-xs px-1 py-0.5 rounded-full";
    switch (difficulty.toLowerCase()) {
      case "easy":
        return `${baseStyles} bg-[#00FF84] border border-[#1AFF00]`;
      case "medium":
        return `${baseStyles} bg-[#F37500] border border-[#FFC084]`;
      case "hard":
        return `${baseStyles} bg-[#FF0000] border border-[#FF8484]`;
      default:
        return `${baseStyles} bg-gray-500 border border-gray-400`;
    }
  };

  const handleDetailsClick = (e) => {
    e.stopPropagation();
    // setSelectedChallenge(challenge);
    navigate("/details",{state:{challenge}});
  };

  return (
    <div 
      className="bg-[#1A0F2B] border-2 border-[#301F4C] rounded-[11px] p-6 cursor-pointer hover:border-[#512E8B] transition-colors"
      onClick={onClick}
    >
      <div className="flex justify-between items-start mb-5">
        <div>
          <h3 className="text-white text-2xl font-medium mb-2.5">{challenge.name}</h3>
          <p className="text-[#CDCDCD] text-lg">
            Complete {challenge.stepGoal} steps
            {challenge.exercises?.length ? ` + ${challenge.exercises.length} exercises` : ''}
          </p>
        </div>
        <span className={getDifficultyStyles(challenge.difficulty)}>
          {challenge.difficulty}
        </span>
      </div>

      <div className="bg-[#403359] rounded-lg px-4 py-[17px]">
        <div className="flex justify-between items-end">
          <div>
            <p className="text-white text-xl mb-4">Stake Range</p>
            <p className="text-white text-lg">Reward Multiplier:</p>
          </div>
          <div className="text-right">
            <p className="text-white text-lg mb-[15px]">
              {challenge.minStake} ETH - {challenge.maxStake} ETH
            </p>
            <p className="text-white text-lg">{challenge.rewardMultiplier}x rewards</p>
          </div>
        </div>
      </div>

      <div className="flex justify-end mt-[38px]">
        <button className="details-btn" onClick={handleDetailsClick}>
          Details
          <ArrowRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

export default ChallengeCard;