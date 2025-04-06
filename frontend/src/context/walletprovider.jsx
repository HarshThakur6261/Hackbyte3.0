import React, { useState } from "react";
import walletcontext from "./walletcontext";

const Walletprovider = ({ children }) => {
  const [account, Setaccount] = useState("Connect Wallet");
  const [balance , Setbalance] = useState("0");

  return (
    <walletcontext.Provider value={{ account, Setaccount , balance , Setbalance }}>
      {children}
    </walletcontext.Provider>
  );
};

export default Walletprovider;
