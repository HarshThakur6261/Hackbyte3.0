import React, { useState } from "react";
import { ArrowRight } from "lucide-react";
import "../styles/DesktopHome.css";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";

const DetailsCardDes = ({
  title,
  description1,
  description2,
  description3,

}) => {


  return (
    <div className="bg-[#1A0F2B] border-2 border-[#301F4C] mb-5 rounded-[11px] p-6">
      <div className="flex justify-between items-start mb-5">
        <div>
          <h3 className="text-white text-2xl font-medium mb-7">{title}</h3>
          <p className="text-[#CDCDCD] text-lg mb-2">{description1}</p>
          <p className="text-[#CDCDCD] text-lg mb-2">{description2}</p>
          <p className="text-[#CDCDCD] text-lg ">{description3}</p>


        </div>
      </div>
      <div className="flex justify-between items-start mb-5">
        <div>
          <h3 className="text-white text-2xl font-medium mb-7">How is it beneficial for you</h3>
          <ul>
            <li>Increased Calorie Burn & Muscle Engagement
Squats and push-ups engage multiple large muscle groups, increasing the intensity of the workout and leading to higher calorie burn compared to walking alone.</li>
<br />
 <li>Muscle Building & Metabolism Boost
These exercises help build lean muscle mass, which boosts your metabolism and allows you to burn more calories even at rest, leading to more efficient weight loss over time.</li>
<br />
<li> Enhanced Body Composition & Afterburn Effect
Squats and push-ups create an afterburn effect (EPOC), keeping your metabolism elevated after exercise. This helps you burn more calories even after the workout ends, while improving body composition by reducing fat and building muscle</li>
          </ul>
          

        </div>
      </div>

    </div>
  );
};

export default DetailsCardDes;
