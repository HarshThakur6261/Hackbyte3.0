import { useState, useEffect } from "react";
import AnimatedNumber from "./AnimatedNumber";
import useFitnessData from "../../Utils/useStepCount";

const ProgressCircle = ({ exercise, handlecompleteExercise }) => {
  const alreadySteps = parseInt(localStorage.getItem("challengeInitialSteps")) || 0;
  const { todaySteps } = useFitnessData();
  const [progress, setProgress] = useState(0);

  const targetSteps = exercise?.reps || 1;
  const circleCircumference = 2 * Math.PI * 54; // More accurate calculation

  const effectiveSteps = Math.max(todaySteps - alreadySteps, 0);
  const progressPercentage = targetSteps > 0 ? Math.min(effectiveSteps / targetSteps, 1) : 0;
  const progressOffset = circleCircumference * (1 - progressPercentage);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setProgress(progressOffset);
    }, 1000);
  
    return () => clearInterval(intervalId);
  }, [todaySteps, progressOffset]);

  useEffect(() => {
    if (progressPercentage >= 1 && exercise && !exercise.isCompleted) {
      handlecompleteExercise(exercise._id);
    }
  }, [progressPercentage, handlecompleteExercise, exercise?._id]);

  const getProgressColor = () => {
    const normalizedProgress = progressPercentage;
    
    if (normalizedProgress < 0.25) {
      // Red to Orange
      const ratio = normalizedProgress / 0.25;
      return `rgb(${Math.floor(255 * (1 - ratio) + 255 * ratio)}, 
              ${Math.floor(59 * (1 - ratio) + 149 * ratio)}, 
              ${Math.floor(48 * (1 - ratio))}`;
    } else if (normalizedProgress < 0.75) {
      // Orange to Yellow
      const ratio = (normalizedProgress - 0.25) / 0.5;
      return `rgb(${Math.floor(255 * (1 - ratio) + 255 * ratio)}, 
              ${Math.floor(149 * (1 - ratio) + 204 * ratio)}, 
              ${Math.floor(0 * (1 - ratio) + 0 * ratio)})`;
    } else {
      // Yellow to Green
      const ratio = (normalizedProgress - 0.75) / 0.25;
      return `rgb(${Math.floor(255 * (1 - ratio) + 76 * ratio)}, 
              ${Math.floor(204 * (1 - ratio) + 217 * ratio)}, 
              ${Math.floor(0 * (1 - ratio) + 100 * ratio)})`;
    }
  };

  return (
    <div className="relative flex items-center justify-center w-[170px] h-[170px]">
      <svg width="175" height="175" viewBox="0 0 120 120">
        {/* Background circle */}
        <circle 
          cx="60" 
          cy="60" 
          r="54" 
          fill="none" 
          stroke="#2B1748" 
          strokeWidth="12" 
        />
        {/* Progress circle */}
        <circle
          cx="60"
          cy="60"
          r="54"
          fill="none"
          stroke={getProgressColor()}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circleCircumference}
          strokeDashoffset={progress}
          transform="rotate(-90 60 60)"
          style={{ 
            transition: "stroke-dashoffset 1.5s ease-out, stroke 0.5s ease-out",
            filter: "drop-shadow(0 0 5px rgba(123, 97, 255, 0.5))"
          }}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center text-center">
        <div className="text-2xl font-bold text-white">
          <AnimatedNumber value={effectiveSteps} />
        </div>
        <div className="text-xs text-purple-300 mt-1">
          of {targetSteps} steps
        </div>
      </div>
    </div>
  );
};

export default ProgressCircle;