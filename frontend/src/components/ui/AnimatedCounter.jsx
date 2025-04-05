import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

const AnimatedCounter = ({ value, duration = 2000, suffix = '', className = '' }) => {
  const [displayValue, setDisplayValue] = useState(0);
  const animationRef = useRef(null);
  const startTimeRef = useRef(null);
  const startValueRef = useRef(0);

  useEffect(() => {
    // Cancel any existing animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    // Store the starting value for the animation
    startValueRef.current = displayValue;
    startTimeRef.current = performance.now();

    const animate = (timestamp) => {
      if (!startTimeRef.current) startTimeRef.current = timestamp;
      
      const progress = Math.min((timestamp - startTimeRef.current) / duration, 1);
      const currentValue = Math.floor(progress * (value - startValueRef.current) + startValueRef.current);
      
      setDisplayValue(currentValue);
      
      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [value, duration]);

  return (
    <span className={className}>
      {displayValue.toLocaleString()}
      {suffix && <span className="counter-suffix">{suffix}</span>}
    </span>
  );
};

AnimatedCounter.propTypes = {
  value: PropTypes.number.isRequired,
  duration: PropTypes.number,
  suffix: PropTypes.string,
  className: PropTypes.string
};

export default AnimatedCounter;