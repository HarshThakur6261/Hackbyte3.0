import React from "react";

const StatsCard = ({ icon, label, value, suffix, className }) => {
  return (
    <div className={`stats-card ${className}`}>
      <img src={icon} alt="" className="stats-card__icon" />
      <div className="stats-card__content">
        <div className="stats-card__label">{label}</div>
        <div className="stats-card__value-container">
          <span className="stats-card__value">{value}</span>
          {suffix && <span className="stats-card__suffix">{suffix}</span>}
        </div>
      </div>
    </div>
  );
};

export default StatsCard;
