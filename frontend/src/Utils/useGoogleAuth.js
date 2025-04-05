// src/hooks/useGoogleAuth.js
import { useEffect } from "react";
import { gapi } from "gapi-script";
import { useAuth } from "../context/AuthContext";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export const useGoogleAuth = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;
  const SCOPES = [
    'https://www.googleapis.com/auth/fitness.activity.read', // For steps
    'https://www.googleapis.com/auth/fitness.body.read'      // For calories
  ].join(' ');
  // Initialize Google API client
  useEffect(() => {
    const initClient = async () => {
      try {
        await gapi.load("client:auth2", async () => {
          await gapi.client.init({
            clientId: CLIENT_ID,
            scope: SCOPES,
            ux_mode: "popup",
          });
        });
      } catch (error) {
        console.error("Error initializing Google API:", error);
      }
    };

    const script = document.createElement("script");
    script.src = "https://apis.google.com/js/api.js";
    script.async = true;
    script.defer = true;
    script.onload = initClient;
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, [CLIENT_ID]);

  // Handle Google Sign-In
  const handleAuth = async () => {
    try {
      const auth2 = gapi.auth2.getAuthInstance();
      const googleUser = await auth2.signIn({ prompt: 'select_account' });
      
      const token = googleUser.getAuthResponse().access_token;
      console.log(token)
     
  
      const { data } = await axios.post('http://localhost:3000/api/users/', {
        email: googleUser.getBasicProfile().getEmail(),
        name: googleUser.getBasicProfile().getName(),
        googleId: googleUser.getBasicProfile().getId()
      });
      
      
      login(token,data.token);
     
  
      // navigate('/home');
    } catch (error) {
      console.error('Authentication error:', error);
    }
  };

  return { handleAuth };
};