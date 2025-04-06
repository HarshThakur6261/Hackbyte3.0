import { useState, useEffect } from "react";
import AnimatedNumber from "./AnimatedNumber";
import useFitnessData from "../../Utils/useStepCount";

const ProgressCircle = ({ exercise, handlecompleteExercise }) => {
  const alreadySteps = parseInt(localStorage.getItem("challengeInitialSteps")) || 0;
  const { todaySteps } = useFitnessData();
  const [progress, setProgress] = useState(0);

  // Safely get target steps with fallback
  const targetSteps = exercise?.reps || 1; // Default to 1 to avoid division by zero
  const circleCircumference = 339.3;

  // Calculate progress with null checks
  const effectiveSteps = Math.max(todaySteps - alreadySteps, 0);
  const progressPercentage = targetSteps > 0 ? Math.min(effectiveSteps / targetSteps, 1) : 0;
  const progressOffset = circleCircumference * (1 - progressPercentage);

  // Rest of the component remains the same...
  const getProgressColor = () => {
    if (progressPercentage < 0.3) return "#FF3B30";
    if (progressPercentage < 0.7) return "#FF9500";
    return "#4CD964";
  };

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

  return (
    <div className="relative flex items-center justify-center w-[170px] h-[170px]">
      <svg width="175" height="175" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="54" fill="none" stroke="#F2F3F1" strokeWidth="12" />
        <circle
          cx="60"
          cy="60"
          r="54"
          fill="none"
          stroke={getProgressColor()}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circleCircumference}
          strokeDashoffset={`${progress}`}
          transform="rotate(-90 60 60)"
          style={{ transition: "stroke-dashoffset 1.5s ease-out" }}
        />
      </svg>
      <div className="absolute text-2xl font-bold text-white">
        <AnimatedNumber value={effectiveSteps} />
      </div>
    </div>
  );
};

export default ProgressCircle;