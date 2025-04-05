import React, { useState, useEffect, useContext } from "react";
import { IoCheckmarkCircle } from "react-icons/io5";
import axios from "axios";
import ProgressCircle from "./ui/ProgressCircle";
import TimerActiveChallenge from "./ui/TimerActiveChallenge";
import useFitnessData from "../Utils/useStepCount";
import getBalance from "../Utils/getbalance";
import Contractcontext from "../context/contractcontext";
import walletcontext from "../context/walletcontext";
import {ethers} from "ethers"

const ActiveChallengeCard = () => {
  const [challengeData, setChallengeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isClaiming, setIsClaiming] = useState(false);
  const {contract} = useContext(Contractcontext);
  const {Setbalance} = useContext(walletcontext);
  const{account} = useContext(walletcontext);
  console.log(challengeData);

  // Fetch active challenge from API
  const fetchActiveChallenge = async () => {
    try {
      const token = localStorage.getItem("JwtToken");
      if (!token) {
        throw new Error("User not authenticated");
      }

      const response = await axios.get(
        `http://localhost:3000/ActiveChallenge/get/${token}`
      );

      if (response.data) {
        setChallengeData(response.data);
      } else {
        throw new Error("No active challenge found");
      }
    } catch (err) {
      if (err.response?.status === 404) {
        setChallengeData(null);
        setError(null);
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchActiveChallenge();
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div className="text-red-500">Error: {error}</div>;
  }

  if (!challengeData) {
    return (
      <div className="flex flex-col grow shrink self-stretch bg-[#1A0F2B] border-4 border-solid border-[#301F4C] rounded-[23px] h-[700px] min-w-[534px] max-md:max-w-full items-center justify-center">
        <div className="text-white text-xl p-8 text-center">
          No active challenge added currently
        </div>
      </div>
    );
  }
  // const handleClaimReward = async () => {
  //   try {
  //     setIsClaiming(true);
  
  //     if (!contract) {
  //       alert("Contract not connected.");
  //       return;
  //     }
  
  //     const tx = await contract.claimReward(); // Call contract method to claim reward
  //     await tx.wait(); // Wait for transaction confirmation
  //     Setbalance(getBalance());
  //     alert("Rewards claimed successfully!");
  
  //     // Now delete the challenge after successful claim
  //     handleDeleteChallenge();
      
  //     // Optionally refresh the challenge data
  //     // fetchActiveChallenge();
      
  //   } catch (error) {
  //     console.error("Error claiming reward:", error);
  //     alert("Failed to claim rewards");
  //   } finally {
  //     setIsClaiming(false);
  //   }
  // };

  const handleClaimReward = async () => {
    // handleDeleteChallenge();
    try {
      setIsClaiming(true);
  
      if (!contract) {
        alert("Contract not connected.");
        return;
      }
  
      // Get the contract balance
      const contractBalance = await contract.getContractBalance();
      const formattedContractBalance = ethers.formatEther(contractBalance);

  // Get the user's reward amount safely
  const userGoal = await contract.getUserGoal(account);
  if (!userGoal || userGoal.length < 2) {
    alert("Failed to fetch user goal.");
    return;
  }

  const [stake, reward] = userGoal;
     
      const formattedReward = ethers.formatEther(reward);
      
      // Check if the contract has enough balance to pay the reward
      if (parseFloat(formattedContractBalance) < parseFloat(formattedReward)) {
        alert("Insufficient contract balance to claim rewards.");
        return;
      }
  
      // Proceed with claiming reward
      const tx = await contract.claimReward();
      await tx.wait(); // Wait for transaction confirmation
  
      const balance = await getBalance(account);
      console.log(balance*156697);
      Setbalance(balance);

      alert("Rewards claimed successfully!");
  
      // Now delete the challenge after successful claim
      handleDeleteChallenge();
  
      // Optionally refresh the challenge data
      // fetchActiveChallenge();
  
    } catch (error) {
      console.error("Error claiming reward:", error);
      alert("Failed to claim rewards");
    } finally {
      setIsClaiming(false);
    }
  };
  
  

  // Define filteredExercises before using it in the JSX
  const filteredExercises = challengeData.exercises.filter(
    (exercise) => exercise.name !== "s"
  );
  const totalExercises = filteredExercises.length;
  const completedExercises = filteredExercises.filter(
    (ex) => ex.isCompleted
  ).length;
  const completionPercentage =
    totalExercises > 0 ? (completedExercises / totalExercises) * 100 : 0;
  const allExercisesCompleted = completedExercises === totalExercises;
  console.log("allExercisesCompleted", allExercisesCompleted);

  const handleDeleteChallenge = async () => {
    try {
      // Retrieve the JWT token from localStorage
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        alert("You need to be logged in to delete the challenge.");
        return;
      }

      // Make the DELETE request using axios
      const response = await axios.delete(
        `http://localhost:3000/ActiveChallenge/delete/${token}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Handle successful response
      console.log("Challenge deleted successfully:", response.data);
      localStorage.removeItem("challengeInitialSteps");
      fetchActiveChallenge();
      alert("Active challenge deleted successfully!");

      // Optionally, update the UI (e.g., redirect the user, refresh data, etc.)
      // For example, you might want to clear the active challenge from the UI
      // or redirect the user to another page
    } catch (error) {
      console.error("Error deleting challenge:", error);
      alert("An error occurred while deleting the active challenge.");
    }
  };

  const handleCompleteExercise = async (exerciseId) => {
    try {
      // Retrieve the token from localStorage
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        return;
      }

      // Define the request body
      const requestBody = {
        exerciseId: exerciseId,
      };

      // Make the PUT request to the API using axios
      const response = await axios.put(
        `http://localhost:3000/ActiveChallenge/update/${token}`,
        requestBody,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      // Handle successful response
      console.log("Exercise marked as completed:", response.data);
      fetchActiveChallenge();
      alert("Exercise completed successfully!");

      // Optionally, you can update the UI or re-fetch the challenge data

      // For example, you can use state or context to refresh the UI
    } catch (error) {
      console.error("Error completing exercise:", error);
      alert("An error occurred while marking the exercise as completed.");
    }
  };

  return (
    <div className="flex flex-col grow shrink self-stretch bg-[#1A0F2B] border-4 border-solid border-[#301F4C] rounded-[23px] h-[700px] min-w-[534px] max-md:max-w-full">
      {/* Header section */}
      <div className="flex flex-col justify-center py-4 pr-10 pl-9 w-full bg-[#2B1748] min-h-[55px] rounded-[22px_22px_0px_0px] max-md:px-5 max-md:max-w-full">
        <div className="flex justify-between items-center max-md:max-w-full">
          <div className="self-stretch my-auto text-xl leading-none text-green-500">
            Active Challenge
          </div>
          <div className="flex gap-2 items-center self-stretch my-auto text-xs leading-6 text-center text-white whitespace-nowrap">
            <div className="flex gap-1.5 items-center self-stretch px-1 py-0.5 my-auto bg-yellow-800 border border-orange-500 border-solid rounded-[37px] w-[45px]">
              <div className="self-stretch my-auto">
                {challengeData.exercises[0]?.completedReps || 0}
              </div>
            </div>
            <div className="gap-1.5 self-stretch px-1 py-0.5 my-auto w-14 bg-blue-900 border border-violet-700 border-solid rounded-[37px]">
              Today
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="self-center flex flex-col flex-1 mt-7 max-w-full">
        {/* Exercises list */}
        <div className="text-lg leading-none text-white max-md:max-w-full">
          {filteredExercises.map((exercise) => (
            <div key={exercise._id}>
              <div className="flex gap-2 items-center">
                {exercise.reps === exercise.completedReps && (
                  <IoCheckmarkCircle className="text-green-500 w-5 h-5" />
                )}
                {exercise.name} ({exercise.reps})
              </div>
            </div>
          ))}
        </div>

        {/* Progress section */}
        <div className="flex gap-9 items-center mt-10 w-full max-md:max-w-full">
          <ProgressCircle
            handlecompleteExercise={handleCompleteExercise}
          
            exercise={challengeData.exercises.find((ex) => ex.name === "steps")}
          />

          <div className="flex flex-col flex-1 shrink justify-center self-stretch">
            <div className="max-w-full w-[231px]">
              <div className="text-sm leading-loose text-white">
                Today's Exercises
              </div>
              {filteredExercises.map((exercise) => (
                <div
                  key={exercise._id}
                  className="flex flex-1 gap-2 items-center pb-1.5 w-[full]"
                >
                  {exercise.reps === exercise.completedReps && (
                    <IoCheckmarkCircle className="text-green-500 w-5 h-5" />
                  )}
                  <div className="self-stretch my-auto text-xs leading-6 text-white">
                    {exercise.name} ({exercise.reps})
                  </div>

                  {exercise.isVideoRequired && (
                    <>
                      <div className="flex flex-1 gap-1 items-center self-stretch my-auto bg-gray-700 min-h-[23px] rounded-[100px] w-full">
                        <div className="flex gap-2.5 items-center self-stretch px-2 py-1 my-auto bg-purple-900 min-h-[23px] rounded-[100px] w-[35px]">
                          <img
                            src="https://cdn.builder.io/api/v1/image/assets/TEMP/33266c09e91dbfd9d462bcbee6a56ad979c4cca1?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
                            className="object-contain self-stretch my-auto aspect-square w-[18px]"
                          />
                        </div>
                        <div className="self-stretch my-auto text-xs leading-6 text-white">
                          Capture
                        </div>
                      </div>
                    </>
                  )}
                  <button
                    onClick={() => {
                      handleCompleteExercise(exercise._id);
                    }}
                  >
                    complete
                  </button>
                </div>
              ))}
            </div>

            <div className="px-3.5 pt-3 pb-3 mt-2.5 w-full text-base leading-none text-white bg-gray-700 rounded-lg min-h-[61px]">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center">
                  <span>{completionPercentage.toFixed(2)}% complete</span>
                </div>
                <div className="w-full h-2 bg-[#FFFFFF] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#CF15E0] rounded-full transition-all duration-300 ease-in-out"
                    style={{ width: `${completionPercentage.toFixed(2)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Stats section */}
        <div className="mt-10 w-full text-white max-md:max-w-full">
          <div className="flex gap-4 items-start max-md:max-w-full">
            <TimerActiveChallenge
              handleDeleteChallenge={handleDeleteChallenge}
              fetchedData={challengeData}
            />

            <div className="flex gap-4 items-center px-7 py-2 whitespace-nowrap bg-purple-900 min-h-[74px] rounded-[140px] w-[225px] max-md:px-5">
              <img
                src="https://cdn.builder.io/api/v1/image/assets/TEMP/b8c47120a8d5f778083924c03a11a3fbcf00948a?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
                className="object-contain shrink-0 self-stretch my-auto aspect-[0.98] rounded-[1030px] w-[52px]"
              />
              <div className="self-stretch my-auto w-[67px]">
                <div className="text-sm leading-loose">Rewards</div>
                <div className="text-2xl">
                  {challengeData.rewardsEarned} ETH
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col justify-center items-start px-2.5 py-2 mt-7 w-full text-sm leading-loose rounded bg-slate-600 min-h-[58px] max-md:max-w-full">
            <div className="flex gap-2.5 items-center">
              <img
                src="https://cdn.builder.io/api/v1/image/assets/TEMP/ff6840f88982558880b258883f520bb119c910b1?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
                className="object-contain shrink-0 self-stretch my-auto aspect-square w-[18px]"
              />
              <div className="self-stretch my-auto w-[77px]">
                <div>ETH Staked</div>
                <div>{challengeData.ethStaked} ETH</div>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-6 mb-6 w-full">
          <button
            onClick={handleClaimReward}
            disabled={!allExercisesCompleted || isClaiming}
            className={`w-full py-3 rounded-lg font-medium text-white transition-colors ${
              allExercisesCompleted
                ? "bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                : "bg-gray-600 cursor-not-allowed"
            }`}
          >
            {isClaiming ? "Claiming..." : "Claim Reward"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ActiveChallengeCard;
