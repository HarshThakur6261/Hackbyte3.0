import React, { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "../styles/DesktopLogin.css";
import { useGoogleAuth } from "../utils/useGoogleAuth";
import { useAuth } from "../context/AuthContext";
import { jwtDecode} from "jwt-decode";

const DesktopLogin = () => {
  const navigate = useNavigate();
  // const location = useLocation();
  const { token ,JwtToken} = useAuth();
  const { handleAuth } = useGoogleAuth();

  // Redirect if already logged in
  useEffect(() => {
    if (token && JwtToken) {
      try {
        const decoded = jwtDecode(JwtToken);
        const email = decoded?.email;
  
        if (email) {
          // Check if the user is new or not from the response
          const isNewUser = decoded?.isNewUser;
  
          if (isNewUser) {
            navigate("/input", {
              state: {
                email: email,
              },
            });
          } else {
            navigate("/");  // Returning user
          }
        }
      } catch (err) {
        console.error("JWT decode error:", err);
      }
    }
  }, [token, JwtToken, navigate]);
  

  return (
    <div className="desktop-login-container">
      <div className="desktop-login-image-section">
        <img
          src="https://cdn.builder.io/api/v1/image/assets/TEMP/af7b5601080f9ca661ace3ac1a1a965a80ae731d?placeholderIfAbsent=true"
          alt=""
          className="desktop-login-hero-image"
        />
      </div>
      <div className="desktop-login-content-section">
        <div className="desktop-login-header">
          <img
            src="https://cdn.builder.io/api/v1/image/assets/TEMP/a301272bd6540764fbb742e1e132906765e3037c?placeholderIfAbsent=true"
            alt=""
            className="desktop-login-logo"
          />
          <div className="desktop-login-brand-name">WeFit</div>
        </div>
        <div className="desktop-login-welcome-section">
          <div className="desktop-login-title">Welcome To WeFit</div>
          <div className="desktop-login-subtitle">Sweat, hustle, and earn</div>
        </div>
        <div className="desktop-login-form-container">
          <div className="desktop-login-button-wrapper">
            <button 
              className="desktop-login-google-button"
              onClick={handleAuth}
            >
              <img
                src="https://cdn.builder.io/api/v1/image/assets/TEMP/554f60cb8c0107cf713b528fea38b3734ddb9b80?placeholderIfAbsent=true"
                alt=""
                className="desktop-login-google-icon"
              />
              <span className="desktop-login-google-text">Continue With Google</span>
            </button>
          </div>
          <div className="desktop-login-terms-text">
            By continuing you agree to our terms and conditions
          </div>
        </div>
        <div className="desktop-login-footer-text">
          Start your fitness journey with Stakefit today
        </div>
      </div>
    </div>
  );
};

export default DesktopLogin;