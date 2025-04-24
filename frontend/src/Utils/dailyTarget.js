// utils/dailyTarget.js
const DEFAULT_TARGETS = {
    steps: 8000,
    calories: 2000,
    distance: 5, // in km
  };
  
  export const getDailyTargets = () => {
    const stored = JSON.parse(localStorage.getItem('dailyTargets'));
    const now = Date.now();
  
    // If no stored target or it's older than 24 hours, reset
    if (!stored || now - stored.timestamp > 24 * 60 * 60 * 1000) {
      const newTargets = {
        ...DEFAULT_TARGETS,
        timestamp: now,
      };
      localStorage.setItem('dailyTargets', JSON.stringify(newTargets));
      return DEFAULT_TARGETS;
    }
  
    return {
      steps: stored.steps,
      calories: stored.calories,
      distance: stored.distance,
    };
  };
  