import { useContext, useEffect, useState } from "react";
import { connectMetaMask } from "../Utils/ConnectWallet";
import { useAuth } from "../context/AuthContext";
import { jwtDecode } from "jwt-decode";
import axios from "axios";
import walletcontext from "../context/walletcontext";

import { connectContract } from "../Utils/ConnectContract";
import Contractcontext from "../context/contractcontext";

const ConnectWallet = () => {
  const { account, Setaccount ,  balance , Setbalance } = useContext(walletcontext);
  const { JwtToken } = useAuth();
  const {Setcontract } = useContext(Contractcontext);

  const truncateAddress = (address) => {
    if (!address || address === "Connect Wallet") return address;
    return `${address.substring(0, 6)}...${address.substring(
      address.length - 4
    )}`;
  };

  const handleconnect = async () => {
    const {account , Balance} = await connectMetaMask();
    Setaccount(account);
    Setbalance(Balance);
    const  contract = await connectContract();
    Setcontract(contract);
    console.log(contract);
  };

  const updateWalletAddress = async () => {
    if (!JwtToken || account === "Connect Wallet") return;

    try {
      const payload = jwtDecode(JwtToken);
      console.log("Decoded JWT:", payload);

      await axios.put(
        `http://localhost:3000/api/users/updateWallet/${payload.email}`,
        {
          walletAddress: account,
        }
      );

      console.log("Wallet address updated");
    } catch (error) {
      console.error("Error updating wallet address:", error);
    }
  };

  useEffect(() => {
    updateWalletAddress();
  }, [account]);

  return (
   <>
    <div className="flex gap-4"> {/* Add gap between the buttons */}
  <button
    onClick={handleconnect}
    className="connect-wallet px-4 py-2 bg-[#512E8B] text-white rounded-full hover:bg-[#3a1d66] transition-colors max-w-[180px] truncate font-mono"
  >
    {truncateAddress(account)}
  </button>
  
  <button
    className="connect-wallet px-4 py-2 bg-[#512E8B] text-white rounded-full hover:bg-[#3a1d66] transition-colors max-w-[180px] truncate font-mono"
  >
    ₹{Math.floor(balance * 156697)}
  </button>
</div>

   </>
    
  );
};

export default ConnectWallet;
