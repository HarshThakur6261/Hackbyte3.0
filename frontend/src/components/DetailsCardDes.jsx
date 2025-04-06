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

    </div>
  );
};

export default DetailsCardDes;
