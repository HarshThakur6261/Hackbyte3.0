// import { ethers } from "ethers";
// import StakeFitABI from "./StakeFitABI.json"

// const CONTRACT_ADDRESS = "0xB49a38E188421F1955B6f86A9477A2B1f3F14A1A";

// export const getEthereumObject = () => {
//   if (window.ethereum) {
//     return window.ethereum;
//   } else {
//     alert("Please install MetaMask to continue!");
//     return null;
//   }
// };

// export const connectContract = async () => {
//   try {
//     const ethereum = getEthereumObject();
//     if (!ethereum) return null;

//     const provider = new ethers.BrowserProvider(ethereum);
//     const signer = await provider.getSigner();
//     const contract = new ethers.Contract(CONTRACT_ADDRESS, StakeFitABI, signer);

//     console.log("Connected to contract:", contract);
//     return contract;
//   } catch (error) {
//     console.error("Error connecting to contract:", error);
//     return null;
//   }
// };


import { ethers } from "ethers";
import StakeFitABI from "./StakeFitABI.json";
//0xC65C63050a442c275AeFba550c9e66588fa8f95f
const CONTRACT_ADDRESS = "0xaCD9D3eB14cf1848457CA21aeB751330b39aA40c";

export const getEthereumObject = () => {
    if (window.ethereum) {
        return window.ethereum;
    } else {
        alert("Please install MetaMask to continue!");
        return null;
    }
};

export const connectContract = async () => {
    try {
        const ethereum = getEthereumObject();
        if (!ethereum) return null;

        const provider = new ethers.BrowserProvider(ethereum);
        const signer = await provider.getSigner(); // Await the getSigner() method
        const contract = new ethers.Contract(CONTRACT_ADDRESS, StakeFitABI, signer);

        console.log("Connected to contract:", contract);
        return contract;
    } catch (error) {
        console.error("Error connecting to contract:", error);
        return null;
    }
};