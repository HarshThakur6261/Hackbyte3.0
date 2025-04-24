import axios from 'axios';

const API_URL = 'http://localhost:3000/api/challenges';

export const getChallenges = async () => {
  return await axios.get(`${API_URL}/get`);
};

export const getChallenge = async (id) => {
  return await axios.get(`${API_URL}/${id}`);
};

export const createChallenge = async (challengeData) => {
  return await axios.post(API_URL, challengeData);
};

export const updateChallenge = async (id, challengeData) => {
  return await axios.patch(`${API_URL}/${id}`, challengeData);
};

export const deleteChallenge = async (id) => {
  return await axios.delete(`${API_URL}/${id}`);
};