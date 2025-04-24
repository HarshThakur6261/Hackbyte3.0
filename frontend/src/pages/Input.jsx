import React, { useState, useEffect, useContext } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { ethers } from 'ethers';
import ContractContext from '../context/ContractContext';
import WalletContext from '../context/WalletContext';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import ConnectWallet from '../components/ConnectWallet';

const Input = () => {
    const location = useLocation();
    const { contract } = useContext(ContractContext);
    const { account } = useContext(WalletContext);
    const email = location.state?.email;
    const navigate = useNavigate();
  
    const [formData, setFormData] = useState({
        height: '',
        weight: '',
        bmi: '',
        age: '',
        gender: '',
        goals: [],
    });
  
    const [formErrors, setFormErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [contractInitialized, setContractInitialized] = useState(false);

    // Initialize contract and check connection
    useEffect(() => {
        if (contract && account) {
            setContractInitialized(true);
        }
    }, [contract, account]);

    // Recalculate BMI whenever height or weight changes
    useEffect(() => {
        calculateBMI(formData.height, formData.weight);
    }, [formData.height, formData.weight]);

    const updateFormData = (updates) => {
        setFormData((prev) => ({ ...prev, ...updates }));
    };

    const validateForm = () => {
        let errors = {};
        
        if (!formData.height) errors.height = "Height is required";
        if (!formData.weight) errors.weight = "Weight is required";
        if (!formData.age) errors.age = "Age is required";
        if (!formData.gender) errors.gender = "Please select a gender";
        if (formData.goals.length === 0) errors.goals = "Please select at least one goal";
        
        setFormErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const isFormValid = () => {
        return (
            formData.height &&
            formData.weight &&
            formData.age &&
            formData.gender &&
            formData.goals.length > 0 &&
            contractInitialized
        );
    };

    const calculateBMI = (height, weight) => {
        const h = parseFloat(height);
        const w = parseFloat(weight);
        if (!isNaN(h) && !isNaN(w) && h > 0 && w > 0) {
            const heightInMeters = h / 100;
            const bmi = (w / (heightInMeters * heightInMeters)).toFixed(1);
            setFormData((prev) => ({ ...prev, bmi }));
        }
    };

    const handleHeightChange = (e) => {
        const value = e.target.value;
        setFormData((prev) => ({ ...prev, height: value }));
    };

    const handleWeightChange = (e) => {
        const value = e.target.value;
        setFormData((prev) => ({ ...prev, weight: value }));
    };

    const goals = [
        {
            id: 'weight-loss',
            title: 'Weight Loss',
            description: 'Reduce body fat and get lean',
        },
        {
            id: 'muscle-gain',
            title: 'Muscle Gain',
            description: 'Build strength and muscle mass',
        },
        {
            id: 'being-fit',
            title: 'Being Fit',
            description: 'Improve stamina and cardio',
        },
        {
            id: 'atheletics-specific',
            title: 'Athletics-Specific',
            description: 'Enhance mobility and flexibility',
        },
    ];

    const toggleGoal = (goalId) => {
        setFormData((prev) => {
            if (prev.goals.includes(goalId)) {
                return { ...prev, goals: prev.goals.filter((g) => g !== goalId) };
            } else if (prev.goals.length < 2) {
                return { ...prev, goals: [...prev.goals, goalId] };
            }
            return prev;
        });
    };

    const handleStake = async () => {
        if (!contract) {
            throw new Error("Smart contract not connected");
        }

        try {
            const weiAmount = ethers.parseEther("0.0065"); // Fixed amount in ETH
            console.log("Sending transaction...");
            
            // Check if the method exists on the contract
            if (!contract.addBalance) {
                throw new Error("addBalance method not found on contract");
            }
            
            const transaction = await contract.addBalance({ value: weiAmount });
            console.log("Transaction hash:", transaction.hash);
            
            await transaction.wait();
            console.log("Transaction confirmed");
            
            // const contractBalance = await contract.getContractBalance();
            // console.log("Contract Balance:", ethers.formatEther(contractBalance));
            
            return true;
        } catch (error) {
            console.error("Staking failed:", error);
            throw error;
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);

        if (!validateForm()) {
            setIsSubmitting(false);
            return;
        }

        try {
            // Check if email exists
            if (!email) {
                throw new Error("Email is missing. Please log in again.");
            }
            
            // Update physical data
            const response = await axios.put(
                `http://localhost:3000/api/users/updatePhysicalData/${email}`, 
                formData
            );
            console.log("Updated user:", response.data);
            
            // Process stake
            await handleStake();
            
            // Show success message
            toast.success("Profile updated and stake processed successfully!");
            
            // Navigate to home
            navigate("/");
            
        } catch (error) {
            console.error("Error:", error);
            toast.error(
                error.response?.data?.message || 
                error.message || 
                "Failed to complete the process"
            );
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <main className="p-8 w-full text-white bg-gray-950 min-h-screen">
         
            <div className="relative mx-auto max-w-[600px]">
            <ConnectWallet/>
                <div className="absolute bg-indigo-700 rounded-full blur-[37.5px] h-[500px] opacity-15 w-[500px]" />
                <div className="absolute right-0 bottom-0 bg-teal-400 rounded-full blur-[37.5px] h-[500px] opacity-15 w-[500px]" />
                <h1 className="mb-8 text-3xl font-bold">Personal Details</h1>
                
                {!email && (
                    <div className="p-4 mb-4 bg-red-900 rounded-lg">
                        <p className="font-medium text-white">Email information is missing. Please log in again.</p>
                    </div>
                )}
                
                {!contractInitialized && (
                    <div className="p-4 mb-4 bg-yellow-900 rounded-lg">
                        <p className="font-medium text-white">Please connect your wallet to continue.</p>
                    </div>
                )}

                <form
                    onSubmit={handleSubmit}
                    className="p-8 bg-gray-900 rounded-xl border border-gray-800 relative z-10"
                >
                    <div className="flex flex-col gap-6">
                        {/* Height & Weight */}
                        <div className="flex gap-4 max-sm:flex-col">
                            <div className="flex-1">
                                <label className="mb-2 block text-sm text-gray-400">Height (cm)</label>
                                <input
                                    type="number"
                                    className={`p-3 w-full text-white bg-gray-800 rounded-lg border ${
                                        formErrors.height ? 'border-red-500' : 'border-gray-700'
                                    }`}
                                    value={formData.height}
                                    onChange={handleHeightChange}
                                    required
                                />
                                {formErrors.height && (
                                    <p className="mt-1 text-xs text-red-500">{formErrors.height}</p>
                                )}
                            </div>
                            <div className="flex-1">
                                <label className="mb-2 block text-sm text-gray-400">Weight (kg)</label>
                                <input
                                    type="number"
                                    className={`p-3 w-full text-white bg-gray-800 rounded-lg border ${
                                        formErrors.weight ? 'border-red-500' : 'border-gray-700'
                                    }`}
                                    value={formData.weight}
                                    onChange={handleWeightChange}
                                    required
                                />
                                {formErrors.weight && (
                                    <p className="mt-1 text-xs text-red-500">{formErrors.weight}</p>
                                )}
                            </div>
                        </div>

                        {/* BMI */}
                        {formData.bmi && (
                            <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                                <span className="text-sm text-gray-400">Your BMI:</span>
                                <span className="ml-2 text-base font-bold">{formData.bmi}</span>
                            </div>
                        )}

                        {/* Age & Gender */}
                        <div className="flex gap-4 max-sm:flex-col">
                            <div className="flex-1">
                                <label className="mb-2 block text-sm text-gray-400">Age</label>
                                <input
                                    type="number"
                                    className={`p-3 w-full text-white bg-gray-800 rounded-lg border ${
                                        formErrors.age ? 'border-red-500' : 'border-gray-700'
                                    }`}
                                    value={formData.age}
                                    onChange={(e) => updateFormData({ age: e.target.value })}
                                    required
                                />
                                {formErrors.age && (
                                    <p className="mt-1 text-xs text-red-500">{formErrors.age}</p>
                                )}
                            </div>
                            <div className="flex-1">
                                <label className="mb-2 block text-sm text-gray-400">Gender</label>
                                <select
                                    className={`p-3 w-full text-white bg-gray-800 rounded-lg border ${
                                        formErrors.gender ? 'border-red-500' : 'border-gray-700'
                                    }`}
                                    value={formData.gender}
                                    onChange={(e) => updateFormData({ gender: e.target.value })}
                                    required
                                >
                                    <option value="">Select gender</option>
                                    <option value="male">Male</option>
                                    <option value="female">Female</option>
                                    <option value="other">Other</option>
                                </select>
                                {formErrors.gender && (
                                    <p className="mt-1 text-xs text-red-500">{formErrors.gender}</p>
                                )}
                            </div>
                        </div>

                        {/* Fitness Goals */}
                        <div>
                            <label className="mb-4 block text-sm text-gray-400">
                                Select up to 2 fitness goals
                            </label>
                            {formData.goals.length === 2 && (
                                <p className="mb-2 text-xs text-yellow-400">
                                    Maximum 2 goals selected. Deselect one to choose another.
                                </p>
                            )}
                            {formErrors.goals && (
                                <p className="mb-2 text-xs text-red-500">{formErrors.goals}</p>
                            )}
                            <div className="grid grid-cols-2 gap-4 max-sm:grid-cols-1">
                                {goals.map((goal) => (
                                    <div
                                        key={goal.id}
                                        className={`rounded-xl border-2 transition-all cursor-pointer ${
                                            formData.goals.includes(goal.id)
                                                ? 'border-purple-500 bg-purple-900'
                                                : formData.goals.length >= 2 && !formData.goals.includes(goal.id)
                                                ? 'border-gray-700 bg-gray-800 opacity-50 cursor-not-allowed'
                                                : 'border-gray-700 bg-gray-800 hover:border-purple-400'
                                        }`}
                                        onClick={() => toggleGoal(goal.id)}
                                    >
                                        <div className="p-5 w-full text-left">
                                            <div className="flex flex-col space-y-2">
                                                <h3
                                                    className={`text-lg font-bold ${
                                                        formData.goals.includes(goal.id)
                                                            ? 'text-white'
                                                            : 'text-gray-200'
                                                    }`}
                                                >
                                                    {goal.title}
                                                </h3>
                                                <p
                                                    className={`text-sm ${
                                                        formData.goals.includes(goal.id)
                                                            ? 'text-purple-200'
                                                            : 'text-gray-400'
                                                    }`}
                                                >
                                                    {goal.description}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            className="py-3 mt-8 w-full font-bold text-white bg-indigo-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-indigo-600 transition-colors"
                            disabled={!isFormValid() || isSubmitting}
                        >
                            {isSubmitting ? "Processing Subscription fees..." : "Continue"}
                        </button>
                    </div>
                </form>
            </div>
        </main>
    );
};

export default Input;