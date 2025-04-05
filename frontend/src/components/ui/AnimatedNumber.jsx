import { useState, useEffect } from "react";

const AnimatedNumber = ({ value }) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let start = 0;
    const duration = 1000; // 1 second
    const increment = Math.ceil(value / 100); // Adjust speed

    const interval = setInterval(() => {
      start += increment;
      if (start >= value) {
        start = value;
        clearInterval(interval);
      }
      setDisplayValue(start);
    }, duration / 100); // Update 100 times in 1 second

    return () => clearInterval(interval);
  }, [value]);

  return <div className="text-3xl font-bold">{displayValue}</div>;
};
export default AnimatedNumber;