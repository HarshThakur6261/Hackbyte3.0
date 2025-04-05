import { useState, useEffect } from "react";
import AnimatedNumber from "./AnimatedNumber";
import useFitnessData from "../../Utils/useStepCount";

const ProgressCircle = ({ exercise, handlecompleteExercise }) => {
  console.log(exercise);
  const alreadySteps = parseInt(localStorage.getItem("challengeInitialSteps")) || 0;
  const targetSteps = exercise.reps ; // Target steps to complete the challenge
  const { todaySteps } = useFitnessData();
  const [progress, setProgress] = useState(0);

  // Total circumference of the circle
  const circleCircumference = 339.3; // 2 * π * r (for r=54)

  // Calculate actual progress after subtracting already completed steps
  const effectiveSteps = Math.max(todaySteps - alreadySteps, 0); // Avoid negative values
  const progressPercentage = Math.min(effectiveSteps / targetSteps, 1); // Normalize within 0-1
  const progressOffset = circleCircumference * (1 - progressPercentage);

  // Dynamic color based on progress
  const getProgressColor = () => {
    if (progressPercentage < 0.3) return "#FF3B30"; // Red (low progress)
    if (progressPercentage < 0.7) return "#FF9500"; // Orange (medium progress)
    return "#4CD964"; // Green (high progress)
  };

  useEffect(() => {
    const intervalId = setInterval(() => {
      console.log(todaySteps); // Logs every second
      setProgress(progressOffset); // Update progress every second
    }, 1000);
  
    return () => clearInterval(intervalId); // Cleanup on unmount/re-render
  }, [todaySteps]);

  useEffect(() => {
    if (progressPercentage >= 1 && !exercise.
      isCompleted) {
      console.log("Challenge Completed");
      
      handlecompleteExercise(exercise._id);
      
     
    }
  }, [progressPercentage, handlecompleteExercise, exercise._id]);

  return (
    <div className="relative flex items-center justify-center w-[90px] h-[90px]">
      <svg width="120" height="120" viewBox="0 0 120 120">
        {/* Background Circle */}
        <circle cx="60" cy="60" r="54" fill="none" stroke="#F2F3F1" strokeWidth="9" />

        {/* Animated Progress Arc */}
        <circle
          cx="60"
          cy="60"
          r="54"
          fill="none"
          stroke={getProgressColor()} // Dynamic color
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={circleCircumference}
          strokeDashoffset={`${progress}`}
          transform="rotate(-90 60 60)"
          style={{ transition: "stroke-dashoffset 1.5s ease-out" }} // Smooth animation
        />
      </svg>

      {/* Animated Number in Center */}
      <div className="absolute text-xl font-bold text-white">
        <AnimatedNumber value={effectiveSteps} />
      </div>
    </div>
  );
};

export default ProgressCircle;
