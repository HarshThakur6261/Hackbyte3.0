import React, { useState, useContext , useEffect } from "react";
import { ArrowRight } from "lucide-react";
import "../styles/DesktopHome.css";
import { Button } from "./ui/button";
import { useChallengeContext } from "../context/ChallengeContext";
import Contractcontext from "../context/contractcontext";
import walletcontext from "../context/walletcontext";
import axios from "axios";
import { Navigate, useNavigate } from "react-router-dom";
import useFitnessData from "../Utils/useStepCount";
import { ethers } from "ethers";
import getBalance from "../Utils/getbalance";

const DetailsCard = ({ challenge }) => {



  const { contract } = useContext(Contractcontext);
  const { account , Setbalance } = useContext(walletcontext);
  const ethToInrRate = 157669;
  const minStake =  0.0006342400852418675;
  const maxStake =  50;
  const rewardMultiplier = challenge?.rewardMultiplier?.toString() || "1.3";
  const [ethAmount, setEthAmount] = useState(minStake.toString());
  const [stakeAmount, setStakeAmount] = useState((minStake * ethToInrRate).toFixed(2));
  const [loading, setLoading] = useState(false);
  const { selectedChallenge } = useChallengeContext();
  const navigate=useNavigate();

  // Handle ETH input
  const handleEthChange = (e) => {
    const value = e.target.value;
    if (value === "" || /^[0-9]*\.?[0-9]*$/.test(value)) {
      setEthAmount(value);
      if (value !== "") {
        setStakeAmount((parseFloat(value || "0") * ethToInrRate).toFixed(2));
      }
    }
  };

  const handleEthBlur = () => {
    let value = parseFloat(ethAmount);
    if (isNaN(value)) value = minStake;
    value = Math.max(minStake, Math.min(maxStake, value));
    setEthAmount(value.toString());
    setStakeAmount((value * ethToInrRate).toFixed(2));
  };

  // Handle INR input
  const handleInrChange = (e) => {
    const value = e.target.value;
    if (value === "" || /^[0-9]*\.?[0-9]*$/.test(value)) {
      setStakeAmount(value);
      if (value !== "") {
        setEthAmount((parseFloat(value || "0") / ethToInrRate).toString());
      }
    }
  };

  const handleInrBlur = () => {
    let value = parseFloat(stakeAmount);
    if (isNaN(value)) value = minStake * ethToInrRate;
    value = Math.max(minStake * ethToInrRate, Math.min(maxStake * ethToInrRate, value));
    setStakeAmount(value.toFixed(2));
    setEthAmount((value / ethToInrRate).toString());
  };

  // Potential reward calculation
  const potentialReward = (
    parseFloat(stakeAmount || "0") * parseFloat(rewardMultiplier.replace("x", "")) +
    parseFloat(stakeAmount || "0")
  ).toFixed(2);



  const { todaySteps } = useFitnessData(); 
  const [currentSteps, setCurrentSteps] = useState(null); // Store the fetched steps

  useEffect(() => {
    if (todaySteps > 0) {
      setCurrentSteps(todaySteps);
    }
  }, [todaySteps]); // Update when `todaySteps` changes

  // // Handle staking function
  // const handleStake = async () => {
  //   if (!account) {
  //     alert("Please connect your wallet first!");
  //     return;
  //   }

  //   setLoading(true);
  //   try {
  //     const stakeData = {
  //       challengeId: challenge._id,
  //       ethStaked: ethAmount,
  //       rewardsEarned: potentialReward,
  //       isCompleted: false
  //     };
      
  //     const response = await axios.post(
  //       `http://localhost:3000/ActiveChallenge/create/${localStorage.getItem('JwtToken')}`,
  //       stakeData
  //     );

  //     if (response.status === 201) {
  //       alert("Challenge started successfully!");
  //       console.log("hyy",todaySteps);
  //       localStorage.setItem("challengeInitialSteps", todaySteps);
  //       navigate("/")
  //     } else {
  //       throw new Error("Failed to create challenge");
  //     }
  //   } catch (error) {
  //     console.error("Staking failed:", error);
  //     alert(`Staking failed: ${error.message}`);
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  // const handleStake = async () => {
  //   console.log("potentialReward",potentialReward);
  //   if (!account) {
  //     alert("Please connect your wallet first!");
  //     return;
  //   }

  //   if (currentSteps === null) {
  //     alert("Fetching step data, please wait...");
  //     return;
  //   }

  //   setLoading(true);
  //   try {
  //     const stakeData = {
  //       challengeId: challenge._id,
  //       ethStaked: ethAmount,
  //       rewardsEarned: potentialReward,
  //       isCompleted: false
  //     };
      
  //     const response = await axios.post(
  //       `http://localhost:3000/ActiveChallenge/create/${localStorage.getItem('JwtToken')}`,
  //       stakeData
  //     );

  //     if (response.status === 201) {
  //       alert("Challenge started successfully!");
  //       localStorage.setItem("challengeInitialSteps", currentSteps); // Store correct steps
  //       navigate("/");
  //     } else {
  //       throw new Error("Failed to create challenge");
  //     }
  //   } catch (error) {
  //     console.error("Staking failed:", error);
  //     alert(`Staking failed: ${error.message}`);
  //   } finally {
  //     setLoading(false);
  //   }
  // };



  const handleStake = async () => {
    console.log("potentialReward", potentialReward);
   
    if (!account) {
        alert("Please connect your wallet first!");
        return;
    }

   

    setLoading(true);
    try {
        // 1. Set goal in the smart contract
        
        const weiAmount = ethers.parseEther(parseFloat(ethAmount).toFixed(18)); // Convert to 18 decimals
        const weiReward = ethers.parseEther(parseFloat(potentialReward/156697).toFixed(18)); // Convert to 18 decimals
        

        // const transaction = await contract.setGoal(weiAmount, weiReward, { value: weiAmount });
         const transaction = await contract.setGoal( weiReward, { value: weiAmount });
        // const transaction = await contract.setGoal({ value: weiAmount });
        // const transaction = await contract.addBalance({ value: weiAmount });


        await transaction.wait(); // Wait for the transaction to be mined


         // 2. Print contract balance
         const contractBalance = await contract.getContractBalance();
         console.log("Contract Balance after setting goal:", ethers.formatEther(contractBalance));
         
         // 3. Print user goal data
         const [stake, reward] = await contract.getUserGoal(account);
         console.log("User stake amount:", ethers.formatEther(stake));
         console.log("User reward amount:", ethers.formatEther(reward));
         
         const balance = await getBalance(account);
         console.log(balance*156697);
         Setbalance(balance);

        // 2. Execute the rest of the function after the contract goal is set
        const stakeData = {
            challengeId: challenge._id,
            ethStaked: ethAmount,
            rewardsEarned: potentialReward,
            isCompleted: false
        };

        const response = await axios.post(
            `http://localhost:3000/ActiveChallenge/create/${localStorage.getItem('JwtToken')}`,
            stakeData
        );

        if (response.status === 201) {
            alert("Challenge started successfully!");
            localStorage.setItem("challengeInitialSteps", currentSteps); // Store correct steps
            navigate("/");
        } else {
            throw new Error("Failed to create challenge");
        }
    } catch (error) {
        console.error("Staking failed:", error);
        alert(`Staking failed: ${error.message}`);
    } finally {
        setLoading(false);
    }
};

  return (
    <div className="bg-[#1A0F2B] border-2 border-[#301F4C] rounded-[11px] p-6">
      <div className="flex justify-between items-start mb-5">
        
      </div>

      <div className="bg-[#403359] rounded-lg px-4 py-[17px] mb-5">
        <div className="mb-4">
          <label className="text-white text-xl block mb-2">Enter Amount (ETH)</label>
          <input
            type="text"
            value={ethAmount}
            onChange={handleEthChange}
            onBlur={handleEthBlur}
            className="w-full bg-[#301F4C] text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
        </div>

        <div className="mb-4">
          <label className="text-white text-xl block mb-2">Stake Amount (INR)</label>
          <input
            type="text"
            value={stakeAmount}
            onChange={handleInrChange}
            onBlur={handleInrBlur}
            className="w-full bg-[#301F4C] text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <p className="text-white text-sm mt-1">
            Minimum amount: ₹{(minStake * ethToInrRate).toFixed(2)} (~{minStake} ETH)
            {maxStake && ` | Maximum amount: ₹${(maxStake * ethToInrRate).toFixed(2)} (~${maxStake} ETH)`}
          </p>
        </div>
      </div>

      <div className="bg-[#403359] rounded-lg px-4 py-[17px] mb-5">
        <div className="flex justify-between items-end">
          <div>
            <p className="text-white text-xl mb-4">Potential Reward</p>
            <p className="text-purple-300 text-sm">Multiplier: {rewardMultiplier}x</p>
          </div>
          <div className="text-right">
            <p className="text-white text-2xl font-bold">₹{potentialReward}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 mb-6">
        <p className="text-black text-sm bg-amber-400 p-2 rounded-lg">
          <span className="font-semibold">Note:</span> Your stake will be refunded if you complete {challenge?.stepGoal?.toLocaleString() || "0"} steps within 24 hours. 
          If you fail, your stake will be distributed to the reward pool.
        </p>
      </div>

      <div className="flex justify-between gap-4 mt-[38px]">
        <button
          className="detailsss-btn"
          onClick={handleStake}
          disabled={loading}
        >
          {loading ? "Staking..." : "Stake Amount"}
        </button>
      </div>
    </div>
  );
};

export default DetailsCard;