export const connectMetaMask = async () => {
  try {
    if (!window.ethereum) {
      console.error("MetaMask is not installed. Please install it to continue.");
      return null;
    }

    // Request accounts
    const accounts = await window.ethereum.request({
      method: 'eth_requestAccounts'
    });

    if (accounts.length > 0) {
      const account = accounts[0];
      console.log("Connected account:", account);

      // Fetch balance
      const balance = await window.ethereum.request({
        method: 'eth_getBalance',
        params: [account, 'latest']
      });

      // Convert balance from Wei to Ether
      const formattedBalance = window.ethers
        ? window.ethers.utils.formatEther(balance)
        : (parseInt(balance, 16) / 1e18).toFixed(6); // Fallback if ethers.js is not available

      console.log("Wallet balance:", formattedBalance, "ETH");

      return { account, Balance: formattedBalance };
    } else {
      console.error("No accounts found.");
      return null;
    }
  } catch (error) {
    console.error("Error connecting to MetaMask:", error);
    return null;
  }
};
