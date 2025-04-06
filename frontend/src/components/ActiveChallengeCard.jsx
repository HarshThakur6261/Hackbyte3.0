import React, { useState, useEffect, useContext } from "react";
import { IoCheckmarkCircle } from "react-icons/io5";
import axios from "axios";
import { ToastContainer, toast } from 'react-toastify';
import ProgressCircle from "./ui/ProgressCircle";
import TimerActiveChallenge from "./ui/TimerActiveChallenge";
import useFitnessData from "../Utils/useStepCount";
import getBalance from "../Utils/getbalance";
import Contractcontext from "../context/contractcontext";
import walletcontext from "../context/walletcontext";
import { ethers } from "ethers"
import { useNavigate } from "react-router-dom";
import { FiCamera, FiDollarSign, FiGift, FiLoader } from 'react-icons/fi';
import { useAuth } from "../context/AuthContext";

const ActiveChallengeCard = ({getUserData}) => {
  const [challengeData, setChallengeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isClaiming, setIsClaiming] = useState(false);
  const { contract } = useContext(Contractcontext);
  const { Setbalance } = useContext(walletcontext);
  const { account } = useContext(walletcontext);
  const navigate = useNavigate();
  const {JwtToken}=useAuth();
  // console.log(challengeData);

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

  const handlecompletechallenge = async () => {
    try {
      // Retrieve the JWT token from localStorage
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        alert("You need to be logged in to delete the challenge.");
        return;
      }

      // Make the DELETE request using axios
      const response = await axios.put(
        `http://localhost:3000/ActiveChallenge/update/challenge/${token}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Handle successful response
      console.log("Challenge completed successfully:", response.data);

      toast.success("Active challenge completed successfully!");

      // Optionally, update the UI (e.g., redirect the user, refresh data, etc.)
      // For example, you might want to clear the active challenge from the UI
      // or redirect the user to another page
    } catch (error) {
      console.error("Error completing challenge:", error);
      alert("An error occurred while completing the active challenge.");
    }
  };

  const handleAddtohistory = async () => {
    try {
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        alert("You need to be logged in to delete the challenge.");
        return;
      }

      // Make the DELETE request using axios
      const response = await axios.post(
        `http://localhost:3000/history/create/${token}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Handle successful response
      console.log("history created  successfully:", response.data);

      toast.success("history created  successfully!");
    } catch (error) {
      console.error("Error history created  challenge:", error);
      alert("An error occurred while history created  .");
    }
  }
  const handleIncreaseChallengeWon = async () => {
    try {
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        return;
      }

      // Make a PUT request to update challengeWon field
      await axios.put(
        `http://localhost:3000/api/users/updateChallengeWon/${token}`, {}, // Assuming this is the route to increment challengeWon

        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

    } catch (error) {
      console.log(error);
    }
  }
  const handleAddRewardsEarned = async () => {
    try {
      const token = localStorage.getItem("JwtToken");

      if (!token) {
        console.error("No JWT token found in localStorage");
        return;
      }

      // Make a PUT request to update challengeWon field
      await axios.put(
        `http://localhost:3000/api/users/updateRewardsEarned/${token}`, {
          rewardAmount:challengeData.rewardsEarned
        }, // Assuming this is the route to increment challengeWon

        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

    } catch (error) {
      console.log(error);

    }
  }

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
      console.log(balance * 156697);
      Setbalance(balance);

      toast.success("Rewards claimed successfully!");


      handlecompletechallenge();
      // Now delete the challenge after successful claim
      handleIncreaseChallengeWon();
      handleAddRewardsEarned();

      handleAddtohistory();

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

  const handleIncreasePoints = async () => {
    try {
      console.log("Attempting to increase points");
      const JwtToken = localStorage.getItem("JwtToken"); // Make sure to get the token
  
      if (!JwtToken) {
        console.error("No JWT token found in localStorage");
        return;
      }
  
      // Make a PUT request with proper headers
      const response = await axios.put(
        `http://localhost:3000/api/users/updatePoints/${JwtToken}`,
        { points: 3 },
        {
          headers: {
            'Authorization': `Bearer ${JwtToken}`,
            'Content-Type': 'application/json'
          }
        }
      );
      getUserData();
      handleAddtohistory();
      handleDeleteChallenge();

      console.log("Points updated successfully:", response.data);
      return response.data;
    } catch (error) {
      console.error("Error increasing points:", error.response?.data || error.message);
      throw error;
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
      toast.success("Active challenge deleted successfully!");

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
      toast.success("Exercise completed successfully!");

      // Optionally, you can update the UI or re-fetch the challenge data

      // For example, you can use state or context to refresh the UI
    } catch (error) {
      console.error("Error completing exercise:", error);
      alert("An error occurred while marking the exercise as completed.");
    }
  };

  const handleCaptureClick = (exerciseId, reps, exerciseName) => {

    // localStorage.setItem("currentExerciseId", exerciseId);
    navigate("/capture", { state: { exerciseId, reps, exerciseName } });
  };

  return (
    <>
    <ToastContainer />

    <div className="flex flex-col grow shrink self-stretch bg-[#1A0F2B] border-4 border-solid border-[#301F4C] rounded-[23px] h-[700px] min-w-[400px] max-md:max-w-full">
      {/* Header section - width unchanged */}
      <div className="flex flex-col justify-center py-4 pr-10 pl-9 w-full bg-gradient-to-r from-[#2B1748] to-[#3A225D] min-h-[55px] rounded-[22px_22px_0px_0px] border-b border-[#4A2D7A] max-md:px-5 max-md:max-w-full">
        <div className="flex justify-between items-center max-md:max-w-full">
          <div className="self-stretch my-auto text-xl font-medium leading-none bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-emerald-300">
            Active Challenge
          </div>
          <div className="flex gap-2 items-center self-stretch my-auto text-xs leading-6 text-center text-white whitespace-nowrap">
            <div className="flex gap-1.5 items-center self-stretch px-2 py-1 my-auto bg-yellow-800/80 border border-orange-500 border-solid rounded-[37px]">
              <div className="self-stretch my-auto font-medium">
                {challengeData.exercises[0]?.completedReps || 0}
              </div>
            </div>
            <div className="gap-1.5 self-stretch px-3 py-1 my-auto bg-blue-900/80 border border-violet-700 border-solid rounded-[37px] font-medium">
              Today
            </div>
          </div>
        </div>
      </div>

      {/* Main content - width unchanged */}
      <div className="self-center flex flex-col flex-1 mt-7 w-full px-6 max-w-full overflow-y-auto">
        {/* Exercises list */}
        <div className="text-lg leading-none text-white max-md:max-w-full mb-6">
          {filteredExercises.map((exercise) => (
            <div key={exercise._id} className="mb-3 p-3 bg-[#2B1748]/50 rounded-lg border border-[#3A225D] hover:bg-[#3A225D]/30 transition-colors">
              <div className="flex gap-2 items-center">
                {exercise.reps === exercise.completedReps ? (
                  <IoCheckmarkCircle className="text-green-400 w-5 h-5" />
                ) : (
                  <div className="w-5 h-5 rounded-full border-2 border-purple-400"></div>
                )}
                <span className="font-medium">
                  {exercise.name} <span className="text-purple-300">({exercise.reps})</span>
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Progress section - width unchanged */}
        <div className="flex gap-6 items-center mt-4 w-full max-md:flex-col max-md:max-w-full">
          <ProgressCircle
            handlecompleteExercise={handleCompleteExercise}
            exercise={challengeData.exercises.find((ex) => ex.name === "steps")}
          />

          <div className="flex flex-col flex-1 shrink justify-center self-stretch w-full">
            <div className="max-w-full w-full">
              <div className="text-sm font-medium leading-loose text-purple-300 mb-2">
                Today's Exercises
              </div>
              {filteredExercises.map((exercise) => (
                <div
                  key={exercise._id}
                  className="flex gap-3 items-center pb-3 w-full"
                >
                  {exercise.reps === exercise.completedReps ? (
                    <IoCheckmarkCircle className="text-green-400 w-5 h-5 flex-shrink-0" />
                  ) : (
                    <div className="w-5 h-5 rounded-full border-2 border-purple-400 flex-shrink-0"></div>
                  )}
                  <div className="self-stretch my-auto text-sm leading-6 text-white font-medium flex-1">
                    {exercise.name} ({exercise.reps})
                  </div>

                  {exercise.isVideoRequired && (
                    <button
                      onClick={() => handleCaptureClick(exercise._id, exercise.reps, exercise.name)}
                      className="flex items-center gap-1 px-3 py-1 bg-purple-900 hover:bg-purple-800 rounded-full text-xs text-white transition-colors"
                    >
                      <FiCamera className="w-3 h-3" />
                      Capture
                    </button>
                  )}
                  <button
                    onClick={() => handleCompleteExercise(exercise._id)}
                    className="text-xs px-3 py-1 bg-blue-900 hover:bg-blue-800 rounded-full text-white transition-colors"
                  >
                    Complete
                  </button>
                </div>
              ))}
            </div>

            <div className="px-3.5 pt-3 pb-3 mt-2.5 w-full text-base leading-none text-white bg-[#2B1748] rounded-lg border border-[#3A225D] min-h-[61px]">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-purple-300">Progress</span>
                  <span>{completionPercentage.toFixed(2)}% complete</span>
                </div>
                <div className="w-full h-2 bg-[#3A225D] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-300 ease-in-out"
                    style={{ width: `${completionPercentage.toFixed(2)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Stats section - width unchanged */}
        <div className="mt-6 w-full text-white max-md:max-w-full">
          <div className="flex gap-4 items-start max-md:flex-col max-md:max-w-full">
            <TimerActiveChallenge
              handleDeleteChallenge={handleDeleteChallenge}
              fetchedData={challengeData}
              className="flex-1"
            />

            <div className="flex gap-4 items-center px-4 py-2 whitespace-nowrap bg-purple-900/80 min-h-[74px] rounded-[140px] border border-[#4A2D7A] w-[225px] max-md:w-full max-md:px-5">
              <div className="bg-[#3A225D] p-2 rounded-full">
                <FiGift className="text-white w-5 h-5" />
              </div>
              <div className="self-stretch my-auto">
                <div className="text-xs text-purple-300">Rewards</div>
                <div className="text-lg font-medium">{challengeData.rewardsEarned} ETH</div>
              </div>
            </div>
          </div>

          <div className="flex flex-col justify-center items-start px-3 py-2 mt-4 w-full text-sm leading-loose rounded-lg bg-[#2B1748] border border-[#3A225D] min-h-[58px] max-md:max-w-full">
            <div className="flex gap-2.5 items-center w-full">
              <div className="bg-[#3A225D] p-1.5 rounded-full">
                <FiDollarSign className="text-white w-3 h-3" />
              </div>
              <div className="self-stretch my-auto">
                <div className="text-xs text-purple-300">ETH Staked</div>
                <div className="text-sm font-medium">{challengeData.ethStaked} ETH</div>
              </div>
            </div>
          </div>
        </div>

        {/* Claim button - width unchanged */}
        <div className="mt-6 mb-6 w-full">
          <button
            onClick={handleIncreasePoints}
            disabled={!allExercisesCompleted || isClaiming}
            className={`w-full py-3 rounded-lg font-medium text-white transition-all ${allExercisesCompleted
              ? "bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 shadow-md"
              : "bg-gray-700 cursor-not-allowed opacity-80"
              }`}
          >
            {isClaiming ? (
              <span className="flex items-center justify-center gap-2">
                <FiLoader className="animate-spin" />
                Claiming...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <FiGift />
               Complete Challenge
              </span>
            )}
          </button>
        </div>
      </div>
    </div>

    </>
  );
};

export default ActiveChallengeCard;
