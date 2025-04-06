import React, { useState } from "react";
import axios from "axios";

const CommunityChallenge = () => {
  const [showModal, setShowModal] = useState(false);
  const [challengeName, setChallengeName] = useState("");
  const [challengeDescription, setChallengeDescription] = useState("");
  const [stakeAmount, setStakeAmount] = useState("");
  const [exercises, setExercises] = useState([
    { name: "", reps: "", sets: "", duration: "", description: "" },
  ]);

  const addExercise = () => {
    setExercises([
      ...exercises,
      { name: "", reps: "", sets: "", duration: "", description: "" },
    ]);
  };

  const updateExercise = (index, field, value) => {
    const updated = [...exercises];
    updated[index][field] = value;
    setExercises(updated);
  };

  const handleCreateChallenge = async () => {
    try {
      // Create the challenge object with string values for all fields
      const newChallenge = {
        communityId: "67f0ba91f400dc2e0589d6fb", // Use actual community ID
        name: challengeName,
        description: challengeDescription,
        exercises: exercises.map(exercise => ({
          ...exercise,
          // Ensure all number-like fields are strings
          reps: exercise.reps.toString(),
          sets: exercise.sets.toString(),
          duration: exercise.duration.toString()
        })),
        stake: { 
          amount: stakeAmount.toString() // Convert to string
        },
      };

      console.log("Submitting challenge:", newChallenge);

      const response = await axios.post(
        "http://localhost:3000/communityChallenge/create", 
        newChallenge
      );

      if (response.status === 200) {
        alert("Challenge created successfully!");
        setShowModal(false);
        // Reset form
        setChallengeName("");
        setChallengeDescription("");
        setStakeAmount("");
        setExercises([{ name: "", reps: "", sets: "", duration: "", description: "" }]);
      } else {
        alert("Failed to create challenge.");
      }
    } catch (error) {
      console.error("Error creating challenge:", error);
      alert("An error occurred while creating the challenge.");
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-4">Community Challenges</h1>

      <button
        onClick={() => setShowModal(true)}
        className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
      >
        + Add Challenge
      </button>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50">
          <div className="bg-white rounded-2xl p-6 w-full max-w-md shadow-xl">
            <h2 className="text-2xl font-semibold mb-4 text-center">
              Create Challenge
            </h2>

            {/* Challenge Name */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">
                Challenge Name*
              </label>
              <input
                type="text"
                value={challengeName}
                onChange={(e) => setChallengeName(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>

            {/* Challenge Description */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">
                Challenge Description*
              </label>
              <textarea
                value={challengeDescription}
                onChange={(e) => setChallengeDescription(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Describe the challenge"
                required
              />
            </div>

            {/* Exercises */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Exercises*</label>
              {exercises.map((exercise, index) => (
                <div key={index} className="flex flex-col gap-2 mb-4 p-3 bg-gray-50 rounded-lg">
                  <input
                    type="text"
                    value={exercise.name}
                    onChange={(e) => updateExercise(index, "name", e.target.value)}
                    placeholder="Exercise Name*"
                    className="w-full px-3 py-2 border rounded-md"
                    required
                  />
                  <input
                    type="text"  // Changed to text input for string reps
                    value={exercise.reps}
                    onChange={(e) => updateExercise(index, "reps", e.target.value)}
                    placeholder="Reps (e.g., '10')"
                    className="w-full px-3 py-2 border rounded-md"
                    pattern="\d*" // Only allow digits
                  />
                  <input
                    type="text"  // Changed to text input for string sets
                    value={exercise.sets}
                    onChange={(e) => updateExercise(index, "sets", e.target.value)}
                    placeholder="Sets (e.g., '3')"
                    className="w-full px-3 py-2 border rounded-md"
                    pattern="\d*" // Only allow digits
                  />
                  <input
                    type="text"  // Changed to text input for string duration
                    value={exercise.duration}
                    onChange={(e) => updateExercise(index, "duration", e.target.value)}
                    placeholder="Duration in seconds (e.g., '60')"
                    className="w-full px-3 py-2 border rounded-md"
                    pattern="\d*" // Only allow digits
                  />
                  <textarea
                    value={exercise.description}
                    onChange={(e) => updateExercise(index, "description", e.target.value)}
                    placeholder="Exercise Description"
                    className="w-full px-3 py-2 border rounded-md"
                  />
                </div>
              ))}
              <button
                type="button"
                onClick={addExercise}
                className="text-blue-500 hover:text-blue-700 text-sm mt-1"
              >
                + Add Exercise
              </button>
            </div>

            {/* Stake Amount */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">
                Stake Amount (ETH)*
              </label>
              <input
                type="text"  // Changed to text input for string amount
                value={stakeAmount}
                onChange={(e) => setStakeAmount(e.target.value.replace(/[^0-9.]/g, ''))}
                placeholder="0.00"
                className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                pattern="^\d*\.?\d*$" // Allow numbers and decimal point
                required
              />
            </div>

            {/* Buttons */}
            <div className="flex justify-between mt-6">
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="bg-gray-300 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-400"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleCreateChallenge}
                disabled={!challengeName || !challengeDescription || !stakeAmount || exercises.some(ex => !ex.name)}
                className={`px-4 py-2 rounded-lg ${
                  !challengeName || !challengeDescription || !stakeAmount || exercises.some(ex => !ex.name)
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-blue-600 text-white hover:bg-blue-700"
                }`}
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CommunityChallenge;