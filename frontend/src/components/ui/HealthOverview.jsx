import React from 'react';
import AnimatedNumber from './AnimatedNumber';
import { IoFootsteps } from "react-icons/io5";
import { FaFire } from "react-icons/fa";
import { RiPinDistanceFill } from "react-icons/ri";

const HealthOverview = () => {
  // Sample data - replace with actual data source when integrating
  const stepsData = { current: 2000, target: 5000 };
  const caloriesData = { current: 90, target: 200 };
  const distanceData = { current: 3.0, target: 5.0 };

  // Progress states - different radius for each circle
  const outerRadius = 54;
  const middleRadius = 42;
  const innerRadius = 30;
  
  // Calculate circumference for each circle
  const outerCircumference = 2 * Math.PI * outerRadius;
  const middleCircumference = 2 * Math.PI * middleRadius;
  const innerCircumference = 2 * Math.PI * innerRadius;

  const getProgressOffset = (current, target, circumference) => {
    const percentage = Math.min(current / target, 1);
    return circumference * (1 - percentage);
  };

  const getPercentage = (current, target) => {
    return Math.min(Math.round((current / target) * 100), 100);
  };

  return (
    <div className="p-4 bg-[#0C0A0B]  rounded-2xl w-[600px]">
      <h2 className="text-xl font-bold text-white mb-4">Health Overview</h2>
      <div className="flex">
      <div className="flex-1 ml-6 flex flex-col justify-center space-y-4">
          {/* Steps Metric */}
          <div className="flex flex-col">
            <div className="flex flex-col ">
              <div className="flex items-center">
                
                <span className="text-xl text-gray-300">Steps</span>
              </div>
              
              <div className="text-white items-center font-light gap-2 text-2xl flex ml-5">
              <IoFootsteps color='#4DBC74' marginRight='10px' />
                
                <AnimatedNumber value={stepsData.current} /> / {stepsData.target}
              </div>
            </div>
            
          </div>
          
          {/* Distance Metric */}
          <div className="flex flex-col">
            <div className="flex  flex-col">
              <div className="flex items-center">
                <span className="text-xl text-gray-300">Distance (km)</span>
              </div>
              <div className="text-white items-center font-light text-2xl gap-2 flex ml-5">
              <RiPinDistanceFill color='#F2BD34' />

                <AnimatedNumber value={distanceData.current} /> / {distanceData.target}
              </div>
            </div>
           
          </div>
          
          {/* Calories Metric */}
          <div className="flex flex-col">
            <div className="flex  flex-col">
              <div className="flex items-center">
                <span className="text-xl text-gray-300">Calories</span>
              </div>
              <div className="text-white items-center font-light gap-2 text-2xl flex ml-5">
              <FaFire color='#FF3B30' />
                <AnimatedNumber value={caloriesData.current} /> / {caloriesData.target}
              </div>
            </div>
            
          </div>
        </div>
        {/* Concentric Circles */}
        <div className="relative flex items-center justify-center w-[180px] h-[180px]">
          <svg width="180" height="180" viewBox="0 0 120 120">
            {/* Distance Circle - Outermost */}
            <circle cx="60" cy="60" r={outerRadius} fill="none" stroke="#44361F" strokeWidth="8" />
            <circle
              cx="60"
              cy="60"
              r={outerRadius}
              fill="none"
              stroke="#FF9500"
              strokeWidth="8"
              strokeDasharray={outerCircumference}
              strokeDashoffset={getProgressOffset(distanceData.current, distanceData.target, outerCircumference)}
              transform="rotate(-90 60 60)"
              style={{ transition: "stroke-dashoffset 1.5s ease-out" }}
            />
            
            {/* Calories Circle - Middle */}
            <circle cx="60" cy="60" r={middleRadius} fill="none" stroke="#3F161E" strokeWidth="8" />
            <circle
              cx="60"
              cy="60"
              r={middleRadius}
              fill="none"
              stroke="#FF3B30"
              strokeWidth="8"
              strokeDasharray={middleCircumference}
              strokeDashoffset={getProgressOffset(caloriesData.current, caloriesData.target, middleCircumference)}
              transform="rotate(-90 60 60)"
              style={{ transition: "stroke-dashoffset 1.5s ease-out" }}
            />
            
            {/* Steps Circle - Innermost */}
            <circle cx="60" cy="60" r={innerRadius} fill="none" stroke="#223B25" strokeWidth="8" />
            <circle
              cx="60"
              cy="60"
              r={innerRadius}
              fill="none"
              stroke="#4CD964"
              strokeWidth="8"
              strokeDasharray={innerCircumference}
              strokeDashoffset={getProgressOffset(stepsData.current, stepsData.target, innerCircumference)}
              transform="rotate(-90 60 60)"
              style={{ transition: "stroke-dashoffset 1.5s ease-out" }}
            />
          </svg>
        </div>
        
        {/* Right Column with Metrics */}
        
      </div>
    </div>
  );
};

export default HealthOverview;
