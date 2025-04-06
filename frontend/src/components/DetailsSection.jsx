import React from "react";
import DetailsCard from "./DetailsCard";
import DetailsCardDes from "./DetailsCardDes";
import { useChallengeContext } from "../context/ChallengeContext";
const DetailsSection = ({ challenge }) => {

  const challengesDes = [
    {
      title: "Challenge Requirement",
      description1: `1. Complete ${challenge.name} in a single day`,
      description2: "2. Excersises must be verified through the app",
      description3: "3. Challenge must be completed within 24 hours",
    },
  ];

  return (
    <div className="w-full max-w-[859px] mx-auto rounded-[19px] bg-[#1A0F2B] border-2 border-[#301F4C] p-8 shadow-lg">
  {/* Challenge Header Section */}
  <div className="mb-10">
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
      <h2 className="text-white text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
        {challenge.name}
      </h2>
      <div className="flex items-center gap-3">
        <span className="text-purple-400 text-sm font-medium px-3 py-1.5 bg-[#301F4C] rounded-full border border-[#3A2C50]">
          {challenge?.type?.toUpperCase() || "CHALLENGE"}
        </span>
        {challenge?.tags?.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {challenge.tags.map((tag, index) => (
              <span 
                key={index} 
                className="text-amber-300 text-xs font-medium px-2.5 py-1 bg-[#3A2C50] rounded-full border border-amber-500/20"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>

    <p className="text-[#CDCDCD] text-lg mb-6">
      Complete this challenge in a single day to earn rewards and get your stake back
    </p>

    {/* Challenge Stats */}
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-[#251838] rounded-xl border border-[#362356]">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-purple-900/50 flex items-center justify-center">
          <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
          </svg>
        </div>
        <div>
          <p className="text-sm text-purple-300">Step Goal</p>
          <p className="text-white font-medium">{challenge?.stepGoal?.toLocaleString() || "0"} steps</p>
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-amber-900/50 flex items-center justify-center">
          <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
          </svg>
        </div>
        <div>
          <p className="text-sm text-amber-300">Difficulty</p>
          <p className="text-white font-medium capitalize">{challenge?.difficulty || "moderate"}</p>
        </div>
      </div>
    </div>

    {/* Exercises Section */}
    <div className="mt-6">
      <h3 className="text-white text-2xl font-bold mb-4">Exercises</h3>
      <ul className="list-disc list-inside text-white">
        {challenge?.exercises?.length > 0 ? (
          challenge.exercises.map((exercise, index) => (
            <li key={index} className="text-lg">{exercise.name} - {exercise.reps} reps</li>
          ))
        ) : (
          <p className="text-gray-400">No exercises available.</p>
        )}
      </ul>
    </div>
  </div>

  {/* Challenge Descriptions */}
  <div className="space-y-6 mb-12">
    {challengesDes.map((challengesDes, index) => (
      <DetailsCardDes
        key={index}
        title={challengesDes.title}
        description1={challengesDes.description1}
        description2={challengesDes.description2}
        description3={challengesDes.description3}
      />
    ))}
  </div>

  {/* Stake Card Section */}
    <DetailsCard challenge={challenge} />
</div>
  );
};

export default DetailsSection;
