import React, { createContext, useContext, useState, useEffect } from 'react';

const ChallengeContext = createContext();

export const ChallengeProvider = ({ children }) => {
  const [challenges, setChallenges] = useState(() => JSON.parse(sessionStorage.getItem('challenges')) || []);
  const [selectedChallenge, setSelectedChallenge] = useState(() => JSON.parse(sessionStorage.getItem('selectedChallenge')) || null);

  useEffect(() => {
    sessionStorage.setItem('challenges', JSON.stringify(challenges));
    localStorage.removeItem('challenges'); // Cleanup
  }, [challenges]);

  useEffect(() => {
    if (selectedChallenge) {
      sessionStorage.setItem('selectedChallenge', JSON.stringify(selectedChallenge));
    } else {
      sessionStorage.removeItem('selectedChallenge');
    }
    localStorage.removeItem('selectedChallenge'); // Cleanup
  }, [selectedChallenge]);

  return (
    <ChallengeContext.Provider value={{ challenges, setChallenges, selectedChallenge, setSelectedChallenge }}>
      {children}
    </ChallengeContext.Provider>
  );
};

export const useChallengeContext = () => useContext(ChallengeContext);
