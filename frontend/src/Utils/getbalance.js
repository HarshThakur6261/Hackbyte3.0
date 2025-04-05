import { ethers } from "ethers";

const getBalance = async (account) => {
  if (!window.ethereum || !account) {
    console.error("Ethereum provider or account not found.");
    return null;
  }

  try {
    const balanceWei = await window.ethereum.request({
      method: "eth_getBalance",
      params: [account, "latest"],
    });

    return ethers.formatEther(balanceWei); // Convert Wei to Ether and return
  } catch (error) {
    console.error("Error fetching balance:", error);
    return null;
  }
};

export default getBalance;
