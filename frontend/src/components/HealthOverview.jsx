import React, { useEffect, useState } from 'react';
import { IoFootsteps } from "react-icons/io5";
import { FaFire } from "react-icons/fa";
import { RiPinDistanceFill } from "react-icons/ri";
import useFitnessData from '../Utils/useStepCount';
import { getDailyTargets } from '../utils/dailyTarget';

const HealthOverview = () => {
  const { todaySteps, todayCalories } = useFitnessData();

  const [targets, setTargets] = useState(getDailyTargets());
  const [progress, setProgress] = useState({
    steps: 0,
    calories: 0,
    distance: 0,
  });

  useEffect(() => {
    // Update progress based on today's data and targets
    const updateProgress = () => {
      setProgress({
        steps: (todaySteps / targets.steps) * 100,
        calories: (todayCalories/ targets.calories) * 100,
        distance: (3 / targets.distance) * 10,  // Assuming 3 km as a placeholder
      });
    };

    updateProgress();
  }, [todaySteps, todayCalories, targets]);

  return (
    <div className="p-6 bg-gradient-to-br from-gray-900 to-gray-950 rounded-2xl w-full shadow-lg border border-gray-800">
      
      <div className="flex flex-row gap-6 items-center">
        {/* Left Column with Metrics */}
        <div className="w-3/5 space-y-4">
          {/* Steps Metric */}
          <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <IoFootsteps className="text-emerald-400" size={18} />
              </div>
              <div className="flex-1">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Steps</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-medium text-white">
                    {todaySteps}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Calories Metric */}
          <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/20">
                <FaFire className="text-red-400" size={18} />
              </div>
              <div className="flex-1">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Calories</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-medium text-white">
                    {Math.floor( todayCalories) }
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Distance Metric */}
          <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                <RiPinDistanceFill className="text-amber-400" size={18} />
              </div>
              <div className="flex-1">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Distance (km)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-medium text-white">
                    {3} {/* Placeholder distance */}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column with Concentric Circles */}
        <div className="relative flex items-center justify-center w-3/5">
          <div className="relative w-55 h-55">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <p className="text-2xl font-bold text-white">
                  {Math.floor(progress.steps)}% 
                </p>
              </div>
            </div>

            <svg width="100%" height="100%" viewBox="0 0 130 130" className="transform -rotate-90">
              {/* Distance Circle - Outermost */}
              <circle cx="65" cy="65" r="54" fill="none" stroke="#3A3A3A" strokeWidth="8" />
              <circle
                cx="65"
                cy="65"
                r="54"
                fill="none"
                stroke="#8B5CF6"
                strokeWidth="8"
                strokeDasharray={2 * Math.PI * 54}
                strokeDashoffset={(1 - progress.distance / 100) * 2 * Math.PI * 54}
                className="transition-all duration-1000 ease-out"
              />

              {/* Calories Circle - Middle */}
              <circle cx="65" cy="65" r="42" fill="none" stroke="#3A3A3A" strokeWidth="8" />
              <circle
                cx="65"
                cy="65"
                r="42"
                fill="none"
                stroke="#F97316"
                strokeWidth="8"
                strokeDasharray={2 * Math.PI * 42}
                strokeDashoffset={(1 - progress.calories / 100) * 2 * Math.PI * 42}
                className="transition-all duration-1000 ease-out"
              />

              {/* Steps Circle - Innermost */}
              <circle cx="65" cy="65" r="30" fill="none" stroke="#3A3A3A" strokeWidth="8" />
              <circle
                cx="65"
                cy="65"
                r="30"
                fill="none"
                stroke="#22D3EE"
                strokeWidth="8"
                strokeDasharray={2 * Math.PI * 30}
                strokeDashoffset={(1 - progress.steps / 100) * 2 * Math.PI * 30}
                className="transition-all duration-1000 ease-out"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HealthOverview;
