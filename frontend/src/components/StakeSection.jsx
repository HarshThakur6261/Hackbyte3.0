import React from "react";

const StakeSection = () => {
  return (
    <div className="stake-section">
      <img
        src="https://cdn.builder.io/api/v1/image/assets/TEMP/bbba16daa3d52e606a5add2f2d28aa205ae94c00?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
        alt=""
        className="stake-section__background"
      />
      <div className="stake-section__content">
        <h2 className="stake-section__title">Stake ETH</h2>
        <div className="stake-section__controls">
          <div className="stake-section__input">
            <span className="stake-section__input-text">Value</span>
            <img
              src="https://cdn.builder.io/api/v1/image/assets/TEMP/5138b04ca8f8b8bb52e8bdeedc115736075abc8a?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
              alt=""
              className="stake-section__input-icon"
            />
          </div>
          <button className="stake-section__button">Collect Wallet</button>
        </div>
      </div>
    </div>
  );
};

export default StakeSection;
