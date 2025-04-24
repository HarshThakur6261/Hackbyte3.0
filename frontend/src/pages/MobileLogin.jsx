import React from "react";
import { useNavigate } from "react-router-dom";
import "../styles/MobileLogin.css";
import { useGoogleAuth } from "../utils/useGoogleAuth";

const MobileLogin = ({ setIsLoggedIn }) => {
  const navigate = useNavigate();
  const { handleAuth } = useGoogleAuth();

  const handleGoogleLogin = async () => {
    console.log("Logging in with Google...");
    await handleAuth(); // Uses the same logic as GoogleAuth.js
    setIsLoggedIn(true); // Optional: Sync with local state if needed
  };

  return (
    <div className="mobile-login">
      <img
        src="https://cdn.builder.io/api/v1/image/assets/TEMP/37c4ae197c41cebc12f189526121111c7d4c1ae1"
        alt="Runner silhouette"
        className="mobile-login__hero-image"
      />
      <div className="mobile-login__content">
        <div className="mobile-login__container">
          <div className="mobile-login__welcome">
            <div className="mobile-login__title">Welcome To WeFit</div>
            <div className="mobile-login__subtitle">
              Sweat, hustle, and earn
            </div>
          </div>
          <div className="mobile-login__form">
            <button 
              className="mobile-login__google-button"
              onClick={handleGoogleLogin}
            >
              <img
                src="https://cdn.builder.io/api/v1/image/assets/TEMP/4f682fc250d802a3e05699a64f855ed2d67bede6"
                alt="Google icon"
                className="mobile-login__google-icon"
              />
              <span className="mobile-login__google-text">
                Continue With Google
              </span>
            </button>
            <div className="mobile-login__terms">
              By continuing you agree to our terms and conditions
            </div>
          </div>
          <div className="mobile-login__footer">
            Start your fitness journey with Stakefit today
          </div>
        </div>
      </div>
    </div>
  );
};

export default MobileLogin;
