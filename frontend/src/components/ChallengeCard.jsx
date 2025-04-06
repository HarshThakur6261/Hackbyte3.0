import React from "react";
import { ArrowRight, Star } from "lucide-react";
import "../styles/DesktopHome.css";
import { useNavigate } from "react-router-dom";

const ChallengeCard = ({ challenge, isRecommended, onClick }) => {
  const navigate = useNavigate();

  const getDifficultyStyles = (difficulty) => {
    const baseStyles = "text-white text-xs px-1 py-0.5 rounded-full";
    switch ((difficulty || '').toLowerCase()) {
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
    navigate("/details", { state: { challenge } });
  };

  return (
    <div 
      className={`bg-[#1A0F2B] border-2 rounded-[11px] p-6 cursor-pointer hover:border-[#512E8B] transition-colors relative ${
        isRecommended 
          ? "border-[#FFD700] shadow-lg shadow-[#FFD700]/20" 
          : "border-[#301F4C]"
      }`}
      onClick={onClick}
    >
      {isRecommended && (
        <div className="absolute top-2 right-2 flex items-center bg-[#FFD700] text-[#1A0F2B] px-2 py-1 rounded-full text-xs font-bold">
          <Star className="w-3 h-3 mr-1" fill="#1A0F2B" />
          AI Recommended
          {challenge.rawData?.RecommendationScore && (
            <span className="ml-1">({challenge.rawData.RecommendationScore.toFixed(1)})</span>
          )}
        </div>
      )}

      <div className="flex justify-between items-start mb-5">
        <div>
          <h3 className="text-white text-2xl font-medium mb-2.5">
            {challenge.name}
          </h3>
          <p className="text-[#CDCDCD] text-lg">
            {challenge.description || 
              `Complete ${challenge.stepGoal || 0} steps${
                challenge.exercises?.length ? ` + ${challenge.exercises.length} exercises` : ''
              }`}
          </p>
        </div>
        <span className={getDifficultyStyles(challenge.difficulty)}>
          {challenge.difficulty || 'Medium'}
        </span>
      </div>

      <div className="bg-[#403359] rounded-lg px-4 py-[17px]">
        <div className="flex justify-between items-end">
          <div>
            {/* <p className="text-white text-xl mb-4">Stake Range</p> */}
            <p className="text-white text-lg">Reward:</p>
          </div>
          <div className="text-right">
            {/* <p className="text-white text-lg mb-[15px]">
              {challenge.minStake || 0.01} ETH - {challenge.maxStake || 0.1} ETH
            </p> */}
            <p className="text-white text-lg">
+5
            </p>
          </div>
        </div>
      </div>

      {isRecommended && challenge.rawData?.Exercises && (
        <div className="mt-4 bg-[#403359]/50 rounded-lg px-4 py-3">
          <p className="text-white text-sm font-medium mb-1">Includes:</p>
          <div className="flex flex-wrap gap-2">
            {challenge.rawData.Exercises.map((exercise, index) => (
              <span key={index} className="text-xs bg-[#512E8B] text-white px-2 py-1 rounded">
                {exercise}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="flex justify-end mt-[38px]">
        <button 
          className="details-btn" 
          onClick={handleDetailsClick}
        >
          Details
          <ArrowRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

export default ChallengeCard;