import Contractcontext from "./contractcontext";
import React, { useState } from "react";
const Contractprovider = ({ children }) => {
  const [contract, Setcontract] = useState(null);

  return (
    <Contractcontext.Provider value={{ contract, Setcontract }}>
      {children}
    </Contractcontext.Provider>
  );
};

export default Contractprovider;
