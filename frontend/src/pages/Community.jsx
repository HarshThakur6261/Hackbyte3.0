// import { useNavigate } from "react-router-dom";
// import React, { useEffect, useState } from "react";
// import ConnectWallet from "../components/ConnectWallet";
// import DesktopChatbot from "../components/DesktopChatbot";
// import "../styles/DesktopHome.css";
// import chatbot from "../assets/chatbot.png";
// import { Avatar } from "../components/ui/avatar";
// import { useAuth } from "../context/AuthContext";
// import { FaHistory, FaCrown, FaUsers } from "react-icons/fa";
// import axios from "axios";

// const Community = () => {
//   const navigate = useNavigate();
//   const [isChatbotOpen, setIsChatbotOpen] = useState(false);
//   const { logout, JwtToken } = useAuth();
//   const [leaderCommunities, setLeaderCommunities] = useState([]);
//   const [memberCommunities, setMemberCommunities] = useState([]);
//   const [otherCommunities, setOtherCommunities] = useState([]);
//   const [showCreateModal, setShowCreateModal] = useState(false);
//   const [showJoinModal, setShowJoinModal] = useState(false);
//   const [allCommunities, setAllCommunities] = useState([]);

//   // Function to navigate to community challenge page
//   const navigateToCommunity = (communityId) => {
//     console.log(communityId);
//     navigate(`/community/add-challenge`, { state: {communityId} });
//   };

//   // Fetch all communities from server
//   const getAllCommunities = async () => {
//     try {
//       const response = await axios.get(
//         `http://localhost:3000/community/new`
//       );
//       setAllCommunities(response.data);
//     } catch (error) {
//       console.log("Error fetching all communities:", error);
//     }
//   };

//   // Fetch user communities and separate them into leader and member communities
//   const getUserCommunities = async () => {
//     try {
//       const response = await axios.get(
//         `http://localhost:3000/community/all/${JwtToken}`,
//         {
//           headers: {
//             Authorization: `Bearer ${JwtToken}`,
//           },
//         }
//       );

//       // Get user ID from auth context or JWT token payload
//       const userId = getUserIdFromToken(JwtToken);

//       // Split communities based on role
//       const leaderComms = [];
//       const memberComms = [];

//       response.data.forEach(community => {
//         // Find the user's member record in the community
//         const userMember = community.members.find(member => 
//           member.userId.toString() === userId.toString() || 
//           member.userId === userId
//         );

//         if (userMember && userMember.role === 'leader') {
//           leaderComms.push(community);
//         } else {
//           memberComms.push(community);
//         }
//       });

//       setLeaderCommunities(leaderComms);
//       setMemberCommunities(memberComms);
//     } catch (error) {
//       console.log("Error fetching user communities:", error);
//     }
//   };

//   // Helper function to extract user ID from JWT token
//   const getUserIdFromToken = (token) => {
//     // Simple implementation - in a real app, you'd decode the token properly
//     try {
//       const base64Url = token.split('.')[1];
//       const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
//       const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
//         return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
//       }).join(''));

//       return JSON.parse(jsonPayload).userId;
//     } catch (e) {
//       console.error('Error decoding token:', e);
//       return null;
//     }
//   };

//   const handleJoinCommunity = async (communityId) => {
//     try {
//       await axios.post(
//         `http://localhost:3000/community/join/${JwtToken}`,
//         { communityId }, // send it in the body
//         {
//           headers: {
//             Authorization: `Bearer ${JwtToken}`,
//           },
//         }
//       );

//       // Refresh community lists
//       await getUserCommunities();
//       updateOtherCommunities();

//     } catch (error) {
//       console.log("Error joining community:", error);
//       alert(error.response?.data?.message || "Failed to join community");
//     }
//   };

//   // Filter out communities that user is already part of
//   const updateOtherCommunities = () => {
//     const userCommunityIds = [...leaderCommunities, ...memberCommunities].map(c => c._id);
//     const others = allCommunities.filter(comm => !userCommunityIds.includes(comm._id));
//     setOtherCommunities(others);
//   };

//   // Load data on component mount
//   useEffect(() => {
//     if (JwtToken) {
//       getAllCommunities();
//       getUserCommunities();
//     }
//   }, [JwtToken]);

//   // Update other communities whenever user or all communities change
//   useEffect(() => {
//     updateOtherCommunities();
//   }, [allCommunities, leaderCommunities, memberCommunities]);

//   const chatMessages = [
//     {
//       sender: "bot",
//       text: "Hi there! I'm your WeFit assistant. How can I help you today?",
//       timestamp: new Date().toLocaleTimeString([], {
//         hour: "2-digit",
//         minute: "2-digit",
//       }),
//     },
//   ];

//   const handleNavigateToHistory = () => {
//     navigate("/history");
//   };

//   const handleToggleChatbot = () => {
//     setIsChatbotOpen(!isChatbotOpen);
//   };

//   // Community Card Component
//   const CommunityCard = ({ community, isLeader, showJoinButton = false }) => {
//     return (
//       <div
//         className={`flex flex-col p-6 rounded-[15px] border-2 border-[#301F4C] bg-[#1A0F2B] hover:border-[#512E8B] transition-colors ${showJoinButton ? '' : 'cursor-pointer'}`}
//         onClick={() => !showJoinButton && navigateToCommunity(community._id)}
//       >
//         <div className="flex justify-between items-start mb-2">
//           <h3 className="text-white text-2xl font-medium">
//             {community.name}
//           </h3>
//           {isLeader && (
//             <div className="flex items-center bg-[#6b44ae] px-2 py-1 rounded-full">
//               <FaCrown className="text-yellow-300 mr-1" />
//               <span className="text-white text-xs">Leader</span>
//             </div>
//           )}
//         </div>

//         <p className="text-gray-300 text-sm mb-3 line-clamp-2">
//           {community.description}
//         </p>

//         <div className="flex flex-wrap gap-2 mb-3">
//           {community.tags && community.tags.map((tag, index) => (
//             <span key={index} className="text-[#CDCDCD] text-sm bg-[#301F4C] px-2 py-1 rounded-full">
//               #{tag}
//             </span>
//           ))}
//         </div>

//         <div className="mt-auto flex items-center justify-between pt-4">
//           <div className="bg-[#301F4C] px-3 py-1 rounded-full flex items-center">
//             <FaUsers className="text-white mr-1" />
//             <span className="text-white text-sm">
//               {community.members?.length || 0} Members
//             </span>
//           </div>

//           {community.completionPercentage > 0 && (
//             <div className="bg-[#301F4C] px-3 py-1 rounded-full">
//               <span className="text-white text-sm">
//                 {community.completionPercentage}% Complete
//               </span>
//             </div>
//           )}

//           {showJoinButton ? (
//             <button 
//               onClick={(e) => {
//                 e.stopPropagation();
//                 handleJoinCommunity(community._id);
//               }}
//               className="px-4 py-2 bg-[#6b44ae] text-white rounded-full hover:bg-[#7b54be]"
//             >
//               Join
//             </button>
//           ) : (
//             <div className="flex items-center justify-center w-8 h-8 rounded-full bg-[#6b44ae]">
//               <span className="text-white font-bold">→</span>
//             </div>
//           )}
//         </div>
//       </div>
//     );
//   };

//   // Community Section Component
//   const CommunitySection = ({ title, communities, isLeader, showJoinButton = false }) => {
//     return (
//       <div className="p-8 rounded-[19px] bg-[#4a336e5c] mb-8">
//         <h2 className="text-white text-3xl font-medium mb-6">
//           {title} ({communities.length})
//         </h2>

//         {communities.length > 0 ? (
//           <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
//             {communities.map((community) => (
//               <CommunityCard 
//                 key={community._id} 
//                 community={community} 
//                 isLeader={isLeader}
//                 showJoinButton={showJoinButton}
//               />
//             ))}
//           </div>
//         ) : (
//           <div className="text-center py-10">
//             <p className="text-gray-300 mb-4">
//               {isLeader ? "You haven't created any communities yet." : 
//                showJoinButton ? "No communities available to join." : "You haven't joined any communities yet."}
//             </p>
//             {!showJoinButton && (
//               <button 
//                 onClick={() => isLeader ? setShowCreateModal(true) : setShowJoinModal(true)}
//                 className="px-6 py-2 bg-[#6b44ae] text-white rounded-full hover:bg-[#7b54be]"
//               >
//                 {isLeader ? "Create a Community" : "Join a Community"}
//               </button>
//             )}
//           </div>
//         )}
//       </div>
//     );
//   };

//   // Create Community Modal
//   const CreateCommunityModal = () => {
//     const [communityData, setCommunityData] = useState({
//       name: "",
//       description: "",
//       tags: [],
//       isPrivate: false,
//       currentTag: ""
//     });

//     const handleChange = (e) => {
//       const { name, value, type, checked } = e.target;
//       setCommunityData(prev => ({
//         ...prev,
//         [name]: type === "checkbox" ? checked : value
//       }));
//     };

//     const addTag = () => {
//       if (communityData.currentTag.trim()) {
//         setCommunityData(prev => ({
//           ...prev,
//           tags: [...prev.tags, prev.currentTag.trim()],
//           currentTag: ""
//         }));
//       }
//     };

//     const removeTag = (indexToRemove) => {
//       setCommunityData(prev => ({
//         ...prev,
//         tags: prev.tags.filter((_, index) => index !== indexToRemove)
//       }));
//     };

//     const handleCreateCommunity = async () => {
//       try {
//         const dataToSubmit = {
//           name: communityData.name,
//           description: communityData.description,
//           tags: communityData.tags,
//         };

//         await axios.post(
//           `http://localhost:3000/community/create/${JwtToken}`,
//           dataToSubmit,
//           {
//             headers: {
//               Authorization: `Bearer ${JwtToken}`,
//             },
//           }
//         );
//         setShowCreateModal(false);
//         navigate('/community/add-challenge');
//         getUserCommunities();

//       } catch (error) {
//         console.log("Error creating community:", error);
//         alert(error.response?.data?.message || "Failed to create community");
//       }
//     };

//     return (
//       <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
//         <div className="bg-[#1A0F2B] border-2 border-[#512E8B] rounded-2xl p-6 w-[500px] max-h-[90vh] overflow-y-auto">
//           <h2 className="text-white text-2xl font-medium mb-4">Create Community</h2>

//           <div className="mb-4">
//             <label className="text-white block mb-2">Community Name*</label>
//             <input
//               type="text"
//               name="name"
//               value={communityData.name}
//               onChange={handleChange}
//               className="w-full bg-[#301F4C] text-white p-2 rounded-lg"
//               placeholder="Enter community name"
//               required
//             />
//           </div>

//           <div className="mb-4">
//             <label className="text-white block mb-2">Description*</label>
//             <textarea
//               name="description"
//               value={communityData.description}
//               onChange={handleChange}
//               className="w-full bg-[#301F4C] text-white p-2 rounded-lg min-h-[100px]"
//               placeholder="Describe your community"
//               required
//             />
//           </div>

//           <div className="mb-4">
//             <label className="text-white block mb-2">Tags</label>
//             <div className="flex">
//               <input
//                 type="text"
//                 name="currentTag"
//                 value={communityData.currentTag}
//                 onChange={handleChange}
//                 className="flex-1 bg-[#301F4C] text-white p-2 rounded-l-lg"
//                 placeholder="Add tags (press Add)"
//               />
//               <button
//                 onClick={addTag}
//                 className="px-4 py-2 bg-[#6b44ae] text-white rounded-r-lg hover:bg-[#7b54be]"
//               >
//                 Add
//               </button>
//             </div>

//             <div className="flex flex-wrap gap-2 mt-2">
//               {communityData.tags.map((tag, index) => (
//                 <div key={index} className="flex items-center gap-1 bg-[#301F4C] text-white px-2 py-1 rounded-full">
//                   <span>#{tag}</span>
//                   <button 
//                     onClick={() => removeTag(index)}
//                     className="text-red-400 hover:text-red-300 font-bold"
//                   >
//                     ×
//                   </button>
//                 </div>
//               ))}
//             </div>
//           </div>

//           <div className="mb-6">
//             <div className="flex items-center mb-2">
//               <input
//                 type="checkbox"
//                 id="isPrivate"
//                 name="isPrivate"
//                 checked={communityData.isPrivate}
//                 onChange={handleChange}
//                 className="mr-2 h-4 w-4"
//               />
//               <label htmlFor="isPrivate" className="text-white">Private Community</label>
//             </div>
//           </div>

//           <div className="flex justify-end gap-4">
//             <button
//               onClick={() => setShowCreateModal(false)}
//               className="px-4 py-2 bg-[#3D2A64] text-white rounded-lg hover:bg-[#4a336e]"
//             >
//               Cancel
//             </button>
//             <button
//               onClick={handleCreateCommunity}
//               disabled={!communityData.name || !communityData.description}
//               className={`px-4 py-2 text-white rounded-lg ${
//                 !communityData.name || !communityData.description 
//                   ? "bg-gray-500 cursor-not-allowed"
//                   : "bg-[#6b44ae] hover:bg-[#7b54be]"
//               }`}
//             >
//               Create
//             </button>
//           </div>
//         </div>
//       </div>
//     );
//   };

//   // Join Community Modal
//   const JoinCommunityModal = () => {
//     const [availableCommunities, setAvailableCommunities] = useState([]);

//     useEffect(() => {
//       const userCommunityIds = [...leaderCommunities, ...memberCommunities].map(c => c._id);
//       const available = allCommunities.filter(
//         c => !userCommunityIds.includes(c._id)
//       );
//       setAvailableCommunities(available);
//     }, [allCommunities, leaderCommunities, memberCommunities]);

//     return (
//       <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
//         <div className="bg-[#1A0F2B] border-2 border-[#512E8B] rounded-2xl p-6 w-[400px]">
//           <h2 className="text-white text-2xl font-medium mb-6">Join Community</h2>
//           {availableCommunities.length === 0 ? (
//             <p className="text-gray-300 mb-6">No communities available to join</p>
//           ) : (
//             <div className="max-h-[300px] overflow-y-auto mb-6">
//               {availableCommunities.map((community) => (
//                 <div 
//                   key={community._id} 
//                   className="flex justify-between items-center p-3 mb-2 rounded-lg bg-[#301F4C] hover:bg-[#3D2A64] cursor-pointer"
//                   onClick={() => handleJoinCommunity(community._id)}
//                 >
//                   <div>
//                     <h3 className="text-white font-medium">{community.name}</h3>
//                     {community.tags && community.tags.length > 0 && (
//                       <p className="text-gray-300 text-sm">#{community.tags[0]}</p>
//                     )}
//                   </div>
//                   <button className="px-3 py-1 bg-[#6b44ae] text-white rounded-full text-sm">
//                     Join
//                   </button>
//                 </div>
//               ))}
//             </div>
//           )}
//           <div className="flex justify-end">
//             <button
//               onClick={() => setShowJoinModal(false)}
//               className="px-4 py-2 bg-[#3D2A64] text-white rounded-lg hover:bg-[#4a336e]"
//             >
//               Close
//             </button>
//           </div>
//         </div>
//       </div>
//     );
//   };

//   return (
//     <div className="desktop-home">
//       <div className="desktop-history__container">
//         <header className="desktop-home__header">
//           <div className="desktop-home__logo-container">
//             <img
//               src="https://cdn.builder.io/api/v1/image/assets/TEMP/69e8365158abd202fc7d010edd0471beda6cd6aa?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
//               alt="Logo"
//               className="desktop-home__logo-image"
//             />
//             <div className="desktop-home__logo-text">StakeFit</div>
//           </div>
//           <div className="flex gap-4 items-center">
//             <div className="overflow-hidden">
//               <ConnectWallet />
//             </div>
//             <button onClick={handleNavigateToHistory} className="rounded-full">
//               <Avatar className="h-[63px] p-3.5 w-[63px] border-4 border-[#512E8B] rounded-full bg-[#413359] cursor-pointer hover:opacity-80 transition-opacity">
//                 <FaHistory color="white" size={30} />
//               </Avatar>
//             </button>
//           </div>
//         </header>

//         <div className="flex flex-col font-poppins p-6">
//           <div className="flex h-[73px] p-3 justify-between items-center rounded-[48px] bg-[#4a336e5c] w-full">
//             <div className="flex items-center gap-2">

//               <button 
//                 className="communitycreate px-6 py-2 bg-[#6b44ae] text-white rounded-full hover:bg-[#7b54be]"
//                 onClick={() => setShowCreateModal(true)}
//               >
//                 Create
//               </button>
//             </div>
//             <div className="text-white font-medium text-lg">
//               Communities ({leaderCommunities.length + memberCommunities.length})
//             </div>
//           </div>

//           <div className="mt-8">
//             {/* Your Communities (Leader) Section */}
//             <CommunitySection 
//               title="Your Communities" 
//               communities={leaderCommunities} 
//               isLeader={true} 
//             />

//             {/* Joined Communities (Member) Section */}
//             <CommunitySection 
//               title="Joined Communities" 
//               communities={memberCommunities} 
//               isLeader={false} 
//             />

//             {/* Other Communities (Available to Join) Section */}
//             <CommunitySection 
//               title="Discover Communities" 
//               communities={otherCommunities} 
//               isLeader={false}
//               showJoinButton={true}
//             />
//           </div>
//         </div>
//       </div>

//       {showCreateModal && <CreateCommunityModal />}
//       {showJoinModal && <JoinCommunityModal />}

//       <div className="chatbot-bubble" onClick={handleToggleChatbot}>
//         <img src={chatbot} alt="Chatbot" />
//       </div>

//       <DesktopChatbot
//         isOpen={isChatbotOpen}
//         onClose={() => setIsChatbotOpen(false)}
//         messages={chatMessages}
//       />
//     </div>
//   );
// };

// export default Community;

// import { useNavigate } from "react-router-dom";
// import React, { useEffect, useState } from "react";
// import ConnectWallet from "../components/ConnectWallet";
// import DesktopChatbot from "../components/DesktopChatbot";
// import "../styles/DesktopHome.css";
// import chatbot from "../assets/chatbot.png";
// import { Avatar } from "../components/ui/avatar";
// import { useAuth } from "../context/AuthContext";
// import { FaHistory, FaCrown, FaUsers, FaPlus, FaSearch } from "react-icons/fa";
// import axios from "axios";

// const Community = () => {
//   const navigate = useNavigate();
//   const [isChatbotOpen, setIsChatbotOpen] = useState(false);
//   const { logout, JwtToken } = useAuth();
//   const [leaderCommunities, setLeaderCommunities] = useState([]);
//   const [memberCommunities, setMemberCommunities] = useState([]);
//   const [otherCommunities, setOtherCommunities] = useState([]);
//   const [showCreateModal, setShowCreateModal] = useState(false);
//   const [showJoinModal, setShowJoinModal] = useState(false);
//   const [allCommunities, setAllCommunities] = useState([]);

//   // Function to navigate to community challenge page
//   const navigateToCommunity = (communityId) => {
//     console.log(communityId);
//     navigate(`/community/add-challenge`, { state: {communityId} });
//   };

//   // Fetch all communities from server
//   const getAllCommunities = async () => {
//     try {
//       const response = await axios.get(
//         `http://localhost:3000/community/new`
//       );
//       setAllCommunities(response.data);
//     } catch (error) {
//       console.log("Error fetching all communities:", error);
//     }
//   };

//   // Fetch user communities and separate them into leader and member communities
//   const getUserCommunities = async () => {
//     try {
//       const response = await axios.get(
//         `http://localhost:3000/community/all/${JwtToken}`,
//         {
//           headers: {
//             Authorization: `Bearer ${JwtToken}`,
//           },
//         }
//       );

//       // Get user ID from auth context or JWT token payload
//       const userId = getUserIdFromToken(JwtToken);

//       // Split communities based on role
//       const leaderComms = [];
//       const memberComms = [];

//       response.data.forEach(community => {
//         // Find the user's member record in the community
//         const userMember = community.members.find(member => 
//           member.userId.toString() === userId.toString() || 
//           member.userId === userId
//         );

//         if (userMember && userMember.role === 'leader') {
//           leaderComms.push(community);
//         } else {
//           memberComms.push(community);
//         }
//       });

//       setLeaderCommunities(leaderComms);
//       setMemberCommunities(memberComms);
//     } catch (error) {
//       console.log("Error fetching user communities:", error);
//     }
//   };

//   // Helper function to extract user ID from JWT token
//   const getUserIdFromToken = (token) => {
//     // Simple implementation - in a real app, you'd decode the token properly
//     try {
//       const base64Url = token.split('.')[1];
//       const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
//       const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
//         return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
//       }).join(''));

//       return JSON.parse(jsonPayload).userId;
//     } catch (e) {
//       console.error('Error decoding token:', e);
//       return null;
//     }
//   };

//   const handleJoinCommunity = async (communityId) => {
//     try {
//       await axios.post(
//         `http://localhost:3000/community/join/${JwtToken}`,
//         { communityId }, // send it in the body
//         {
//           headers: {
//             Authorization: `Bearer ${JwtToken}`,
//           },
//         }
//       );

//       // Refresh community lists
//       await getUserCommunities();
//       updateOtherCommunities();

//     } catch (error) {
//       console.log("Error joining community:", error);
//       alert(error.response?.data?.message || "Failed to join community");
//     }
//   };

//   // Filter out communities that user is already part of
//   const updateOtherCommunities = () => {
//     const userCommunityIds = [...leaderCommunities, ...memberCommunities].map(c => c._id);
//     const others = allCommunities.filter(comm => !userCommunityIds.includes(comm._id));
//     setOtherCommunities(others);
//   };

//   // Load data on component mount
//   useEffect(() => {
//     if (JwtToken) {
//       getAllCommunities();
//       getUserCommunities();
//     }
//   }, [JwtToken]);

//   // Update other communities whenever user or all communities change
//   useEffect(() => {
//     updateOtherCommunities();
//   }, [allCommunities, leaderCommunities, memberCommunities]);

//   const chatMessages = [
//     {
//       sender: "bot",
//       text: "Hi there! I'm your WeFit assistant. How can I help you today?",
//       timestamp: new Date().toLocaleTimeString([], {
//         hour: "2-digit",
//         minute: "2-digit",
//       }),
//     },
//   ];

//   const handleNavigateToHistory = () => {
//     navigate("/history");
//   };

//   const handleToggleChatbot = () => {
//     setIsChatbotOpen(!isChatbotOpen);
//   };

//   const CommunityCard = ({ community, isLeader, showJoinButton = false }) => {
//     return (
//       <div
//         className={`relative flex flex-col p-6 rounded-xl border-2 border-[#3A225D] bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] hover:border-[#6B44AE] transition-all duration-300 shadow-lg hover:shadow-purple-500/20 ${showJoinButton ? '' : 'cursor-pointer group'}`}
//         onClick={() => !showJoinButton && navigateToCommunity(community._id)}
//       >
//         {/* Glow effect on hover */}
//         <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 bg-gradient-to-br from-purple-500/10 to-transparent transition-opacity duration-300 pointer-events-none"></div>

//         <div className="flex justify-between items-start mb-3">
//           <h3 className="text-white text-xl font-semibold tracking-tight">
//             {community.name}
//           </h3>
//           {isLeader && (
//             <div className="flex items-center bg-gradient-to-r from-yellow-500 to-yellow-600 px-3 py-1 rounded-full shadow-md">
//               <FaCrown className="text-white mr-1 text-sm" />
//               <span className="text-white text-xs font-medium">Leader</span>
//             </div>
//           )}
//         </div>

//         <p className="text-gray-300 text-sm mb-4 line-clamp-2 leading-relaxed">
//           {community.description}
//         </p>

//         <div className="flex flex-wrap gap-2 mb-4">
//           {community.tags?.map((tag, index) => (
//             <span 
//               key={index} 
//               className="text-[#CDCDCD] text-xs bg-[#3A225D] px-3 py-1 rounded-full border border-[#4A2D7A]"
//             >
//               #{tag}
//             </span>
//           ))}
//         </div>

//         <div className="mt-auto flex items-center justify-between pt-4 border-t border-[#3A225D]">
//           <div className="flex items-center bg-[#3A225D] px-3 py-1 rounded-full border border-[#4A2D7A]">
//             <FaUsers className="text-purple-300 mr-2 text-sm" />
//             <span className="text-white text-xs font-medium">
//               {community.members?.length || 0} Members
//             </span>
//           </div>

//           {showJoinButton ? (
//             <button 
//               onClick={(e) => {
//                 e.stopPropagation();
//                 handleJoinCommunity(community._id);
//               }}
//               className="px-4 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all shadow-md text-sm font-medium"
//             >
//               Join Community
//             </button>
//           ) : (
//             <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] group-hover:from-[#7B54BE] group-hover:to-[#9A6FE3] transition-all">
//               <span className="text-white font-bold text-sm">→</span>
//             </div>
//           )}
//         </div>
//       </div>
//     );
//   };

//   // Enhanced CommunitySection component
//   const CommunitySection = ({ title, communities, isLeader, showJoinButton = false }) => {
//     return (
//       <div className="p-8 rounded-2xl bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border border-[#3A225D] mb-8 shadow-lg">
//         <div className="flex justify-between items-center mb-6">
//           <h2 className="text-white text-2xl font-semibold tracking-tight">
//             {title} <span className="text-purple-300 ml-2">({communities.length})</span>
//           </h2>

//           {!showJoinButton && (
//             <button 
//               onClick={() => isLeader ? setShowCreateModal(true) : setShowJoinModal(true)}
//               className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all text-sm font-medium"
//             >
//               <FaPlus className="text-sm" />
//               {isLeader ? "New Community" : "Join Community"}
//             </button>
//           )}
//         </div>

//         {communities.length > 0 ? (
//           <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
//             {communities.map((community) => (
//               <CommunityCard 
//                 key={community._id} 
//                 community={community} 
//                 isLeader={isLeader}
//                 showJoinButton={showJoinButton}
//               />
//             ))}
//           </div>
//         ) : (
//           <div className="text-center py-10">
//             <div className="inline-block p-6 rounded-xl bg-[#2B1748] border border-[#3A225D]">
//               <p className="text-gray-300 mb-4 text-sm">
//                 {isLeader ? "You haven't created any communities yet." : 
//                 showJoinButton ? "No communities available to join." : "You haven't joined any communities yet."}
//               </p>
//               {!showJoinButton && (
//                 <button 
//                   onClick={() => isLeader ? setShowCreateModal(true) : setShowJoinModal(true)}
//                   className="px-6 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all text-sm font-medium"
//                 >
//                   {isLeader ? "Create Your First Community" : "Browse Communities"}
//                 </button>
//               )}
//             </div>
//           </div>
//         )}
//       </div>
//     );
//   };

//   // Enhanced CreateCommunityModal
//   const CreateCommunityModal = () => {
//     const [communityData, setCommunityData] = useState({
//             name: "",
//             description: "",
//             tags: [],
//             isPrivate: false,
//             currentTag: ""
//           });

//           const handleChange = (e) => {
//             const { name, value, type, checked } = e.target;
//             setCommunityData(prev => ({
//               ...prev,
//               [name]: type === "checkbox" ? checked : value
//             }));
//           };

//           const addTag = () => {
//             if (communityData.currentTag.trim()) {
//               setCommunityData(prev => ({
//                 ...prev,
//                 tags: [...prev.tags, prev.currentTag.trim()],
//                 currentTag: ""
//               }));
//             }
//           };

//           const removeTag = (indexToRemove) => {
//             setCommunityData(prev => ({
//               ...prev,
//               tags: prev.tags.filter((_, index) => index !== indexToRemove)
//             }));
//           };

//           const handleCreateCommunity = async () => {
//             try {
//               const dataToSubmit = {
//                 name: communityData.name,
//                 description: communityData.description,
//                 tags: communityData.tags,
//               };

//               await axios.post(
//                 `http://localhost:3000/community/create/${JwtToken}`,
//                 dataToSubmit,
//                 {
//                   headers: {
//                     Authorization: `Bearer ${JwtToken}`,
//                   },
//                 }
//               );
//               setShowCreateModal(false);
//               navigate('/community/add-challenge');
//               getUserCommunities();

//             } catch (error) {
//               console.log("Error creating community:", error);
//               alert(error.response?.data?.message || "Failed to create community");
//             }
//           };

//     return (
//       <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
//         <div className="bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border-2 border-[#6B44AE] rounded-2xl p-8 w-[500px] max-h-[90vh] overflow-y-auto shadow-xl">
//           <h2 className="text-white text-2xl font-semibold mb-6 tracking-tight">Create Community</h2>

//           <div className="space-y-5">
//             <div>
//               <label className="text-purple-300 text-sm font-medium block mb-2">Community Name*</label>
//               <input
//                 type="text"
//                 name="name"
//                 value={communityData.name}
//                 onChange={handleChange}
//                 className="w-full bg-[#2B1748] text-white p-3 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all"
//                 placeholder="Enter community name"
//                 required
//               />
//             </div>

//             <div>
//               <label className="text-purple-300 text-sm font-medium block mb-2">Description*</label>
//               <textarea
//                 name="description"
//                 value={communityData.description}
//                 onChange={handleChange}
//                 className="w-full bg-[#2B1748] text-white p-3 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all min-h-[120px]"
//                 placeholder="Describe your community"
//                 required
//               />
//             </div>

//             <div>
//               <label className="text-purple-300 text-sm font-medium block mb-2">Tags</label>
//               <div className="flex gap-2">
//                 <input
//                   type="text"
//                   name="currentTag"
//                   value={communityData.currentTag}
//                   onChange={handleChange}
//                   onKeyDown={(e) => e.key === 'Enter' && addTag()}
//                   className="flex-1 bg-[#2B1748] text-white p-3 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all"
//                   placeholder="Add tags (press Enter or Add)"
//                 />
//                 <button
//                   onClick={addTag}
//                   className="px-4 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-lg hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all flex items-center justify-center"
//                 >
//                   Add
//                 </button>
//               </div>

//               <div className="flex flex-wrap gap-2 mt-3">
//                 {communityData.tags.map((tag, index) => (
//                   <div key={index} className="flex items-center gap-1 bg-[#3A225D] text-white px-3 py-1 rounded-full border border-[#4A2D7A]">
//                     <span className="text-xs">#{tag}</span>
//                     <button 
//                       onClick={() => removeTag(index)}
//                       className="text-red-400 hover:text-red-300 text-xs"
//                     >
//                       ×
//                     </button>
//                   </div>
//                 ))}
//               </div>
//             </div>

//             <div className="flex items-center">
//               <input
//                 type="checkbox"
//                 id="isPrivate"
//                 name="isPrivate"
//                 checked={communityData.isPrivate}
//                 onChange={handleChange}
//                 className="h-4 w-4 rounded border-[#3A225D] bg-[#2B1748] text-[#6B44AE] focus:ring-[#6B44AE]"
//               />
//               <label htmlFor="isPrivate" className="text-gray-300 ml-2 text-sm">Private Community</label>
//             </div>
//           </div>

//           <div className="flex justify-end gap-3 mt-8">
//             <button
//               onClick={() => setShowCreateModal(false)}
//               className="px-5 py-2 bg-[#3A225D] text-white rounded-lg hover:bg-[#4A2D7A] transition-all border border-[#4A2D7A] text-sm font-medium"
//             >
//               Cancel
//             </button>
//             <button
//               onClick={handleCreateCommunity}
//               disabled={!communityData.name || !communityData.description}
//               className={`px-5 py-2 text-white rounded-lg text-sm font-medium ${
//                 !communityData.name || !communityData.description 
//                   ? "bg-gray-600 cursor-not-allowed"
//                   : "bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all"
//               }`}
//             >
//               Create Community
//             </button>
//           </div>
//         </div>
//       </div>
//     );
//   };

//   // Enhanced JoinCommunityModal
//   const JoinCommunityModal = () => {
//     // ... [keep all existing modal logic exactly the same]
//     const [availableCommunities, setAvailableCommunities] = useState([]);

//         useEffect(() => {
//           const userCommunityIds = [...leaderCommunities, ...memberCommunities].map(c => c._id);
//           const available = allCommunities.filter(
//             c => !userCommunityIds.includes(c._id)
//           );
//           setAvailableCommunities(available);
//         }, [allCommunities, leaderCommunities, memberCommunities]);

//     return (
//       <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
//         <div className="bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border-2 border-[#6B44AE] rounded-2xl p-8 w-[450px] max-h-[80vh] overflow-hidden">
//           <div className="flex justify-between items-center mb-6">
//             <h2 className="text-white text-2xl font-semibold tracking-tight">Join Community</h2>
//             <button
//               onClick={() => setShowJoinModal(false)}
//               className="text-gray-400 hover:text-white transition-colors"
//             >
//               ✕
//             </button>
//           </div>

//           <div className="relative mb-6">
//             <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
//               <FaSearch className="text-gray-400" />
//             </div>
//             <input
//               type="text"
//               placeholder="Search communities..."
//               className="w-full bg-[#2B1748] text-white pl-10 pr-4 py-2 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all"
//             />
//           </div>

//           {availableCommunities.length === 0 ? (
//             <div className="text-center py-8">
//               <p className="text-gray-400 mb-4">No communities available to join</p>
//               <button
//                 onClick={() => setShowCreateModal(true)}
//                 className="px-4 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all text-sm font-medium"
//               >
//                 Create New Community
//               </button>
//             </div>
//           ) : (
//             <div className="space-y-3 max-h-[50vh] overflow-y-auto pr-2">
//               {availableCommunities.map((community) => (
//                 <div 
//                   key={community._id} 
//                   className="flex justify-between items-center p-4 rounded-xl bg-[#2B1748] border border-[#3A225D] hover:bg-[#3A225D] transition-colors cursor-pointer"
//                   onClick={() => handleJoinCommunity(community._id)}
//                 >
//                   <div>
//                     <h3 className="text-white font-medium">{community.name}</h3>
//                     {community.tags?.length > 0 && (
//                       <p className="text-gray-400 text-sm mt-1">
//                         {community.tags.map(tag => `#${tag}`).join(' ')}
//                       </p>
//                     )}
//                   </div>
//                   <button className="px-3 py-1 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full text-xs font-medium hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all">
//                     Join
//                   </button>
//                 </div>
//               ))}
//             </div>
//           )}
//         </div>
//       </div>
//     );
//   };
//   const LeaderboardWidget = () => {
//     // Static leaderboard data for demonstration
//     const leaderboardData = [
//       { rank: 1, name: "Alex Johnson", score: 2450, avatar: "A" },
//       { rank: 2, name: "Maya Chen", score: 2320, avatar: "M" },
//       { rank: 3, name: "Tyler Smith", score: 2105, avatar: "T" },
//       { rank: 4, name: "Sarah Adams", score: 1950, avatar: "S" },
//       { rank: 5, name: "Ryan Park", score: 1820, avatar: "R" }
//     ];

//   return (
//     <div className="min-h-screen bg-[#0F0A1A]">
//       {/* Header remains the same */}
//       <header className="desktop-home__header bg-[#1A0F2B] border-b border-[#3A225D]">

//         <div className="desktop-home__logo-container">
//            <img
//               src="https://cdn.builder.io/api/v1/image/assets/TEMP/69e8365158abd202fc7d010edd0471beda6cd6aa?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
//               alt="Logo"
//               className="desktop-home__logo-image"
//             />
//             <div className="desktop-home__logo-text">WeFit</div>
//           </div>
//           <div className="flex gap-4 items-center">
//             <div className="overflow-hidden">
//               <ConnectWallet />
//             </div>
//             <button onClick={handleNavigateToHistory} className="rounded-full">
//               <Avatar className="h-[63px] p-3.5 w-[63px] border-4 border-[#512E8B] rounded-full bg-[#413359] cursor-pointer hover:opacity-80 transition-opacity">
//                 <FaHistory color="white" size={30} />
//               </Avatar>
//             </button>
//           </div>

//       </header>

//       {/* Main content with enhanced gradient background */}
//       <main className="p-6 bg-gradient-to-br from-[#0F0A1A] to-[#1A0F2B] min-h-[calc(100vh-80px)]">
//         {/* Enhanced top bar */}
//         <div className="flex h-16 p-1 justify-between items-center rounded-full bg-[#2B1748] border border-[#3A225D] shadow-lg mb-8">
//           <div className="flex items-center gap-3 ml-4">
//             <button 
//               className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all shadow-md text-sm font-medium"
//               onClick={() => setShowCreateModal(true)}
//             >
//               <FaPlus className="text-sm" />
//               Create Community
//             </button>
//           </div>
//           <div className="text-white font-medium text-lg bg-[#3A225D] px-6 py-2 rounded-full border border-[#4A2D7A]">
//             My Communities ({leaderCommunities.length + memberCommunities.length})
//           </div>
//           <div className="w-10"></div> {/* Spacer for balance */}
//         </div>

//         {/* Community sections with subtle animations */}
//         <div className="space-y-8">
//           <CommunitySection 
//             title="Your Communities" 
//             communities={leaderCommunities} 
//             isLeader={true} 
//           />

//           <CommunitySection 
//             title="Joined Communities" 
//             communities={memberCommunities} 
//             isLeader={false} 
//           />

//           <CommunitySection 
//             title="Discover Communities" 
//             communities={otherCommunities} 
//             isLeader={false}
//             showJoinButton={true}
//           />
//         </div>
//       </main>

//       {/* Modals */}
//       {showCreateModal && <CreateCommunityModal />}
//       {showJoinModal && <JoinCommunityModal />}

//       {/* Floating elements */}
//       <div className="fixed bottom-8 right-8 flex flex-col gap-4">
//         <div 
//           className="chatbot-bubble w-14 h-14 bg-gradient-to-br from-[#6B44AE] to-[#8A5FD3] hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all shadow-lg"
//           onClick={handleToggleChatbot}
//         >
//           <img src={chatbot} alt="Chatbot" className="p-3" />
//         </div>
//       </div>

//       <DesktopChatbot
//         isOpen={isChatbotOpen}
//         onClose={() => setIsChatbotOpen(false)}
//         messages={chatMessages}
//       />
//     </div>
//   );
// };

// export default Community;

import { useNavigate } from "react-router-dom";
import React, { useEffect, useState } from "react";
import ConnectWallet from "../components/ConnectWallet";
import DesktopChatbot from "../components/DesktopChatbot";
import "../styles/DesktopHome.css";
import chatbot from "../assets/chatbot.png";
import { Avatar } from "../components/ui/avatar";
import { useAuth } from "../context/AuthContext";
import { FaHistory, FaCrown, FaUsers, FaPlus, FaSearch, FaTrophy } from "react-icons/fa";
import axios from "axios";

const Community = () => {
  const navigate = useNavigate();
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const { logout, JwtToken } = useAuth();
  const [leaderCommunities, setLeaderCommunities] = useState([]);
  const [memberCommunities, setMemberCommunities] = useState([]);
  const [otherCommunities, setOtherCommunities] = useState([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showJoinModal, setShowJoinModal] = useState(false);
  const [allCommunities, setAllCommunities] = useState([]);

  // Function to navigate to community challenge page
  const navigateToCommunity = (communityId) => {
    console.log(communityId);
    navigate(`/community/add-challenge`, { state: { communityId } });
  };

  // Fetch all communities from server
  const getAllCommunities = async () => {
    try {
      const response = await axios.get(
        `http://localhost:3000/community/new`
      );
      setAllCommunities(response.data);
    } catch (error) {
      console.log("Error fetching all communities:", error);
    }
  };

  // Fetch user communities and separate them into leader and member communities
  const getUserCommunities = async () => {
    try {
      const response = await axios.get(
        `http://localhost:3000/community/all/${JwtToken}`,
        {
          headers: {
            Authorization: `Bearer ${JwtToken}`,
          },
        }
      );

      // Get user ID from auth context or JWT token payload
      const userId = getUserIdFromToken(JwtToken);

      // Split communities based on role
      const leaderComms = [];
      const memberComms = [];

      response.data.forEach(community => {
        // Find the user's member record in the community
        const userMember = community.members.find(member =>
          member.userId.toString() === userId.toString() ||
          member.userId === userId
        );

        if (userMember && userMember.role === 'leader') {
          leaderComms.push(community);
        } else {
          memberComms.push(community);
        }
      });

      setLeaderCommunities(leaderComms);
      setMemberCommunities(memberComms);
    } catch (error) {
      console.log("Error fetching user communities:", error);
    }
  };

  // Helper function to extract user ID from JWT token
  const getUserIdFromToken = (token) => {
    // Simple implementation - in a real app, you'd decode the token properly
    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(atob(base64).split('').map(function (c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
      }).join(''));

      return JSON.parse(jsonPayload).userId;
    } catch (e) {
      console.error('Error decoding token:', e);
      return null;
    }
  };

  const handleJoinCommunity = async (communityId) => {
    try {
      await axios.post(
        `http://localhost:3000/community/join/${JwtToken}`,
        { communityId }, // send it in the body
        {
          headers: {
            Authorization: `Bearer ${JwtToken}`,
          },
        }
      );

      // Refresh community lists
      await getUserCommunities();
      updateOtherCommunities();

    } catch (error) {
      console.log("Error joining community:", error);
      alert(error.response?.data?.message || "Failed to join community");
    }
  };

  // Filter out communities that user is already part of
  const updateOtherCommunities = () => {
    const userCommunityIds = [...leaderCommunities, ...memberCommunities].map(c => c._id);
    const others = allCommunities.filter(comm => !userCommunityIds.includes(comm._id));
    setOtherCommunities(others);
  };

  // Load data on component mount
  useEffect(() => {
    if (JwtToken) {
      getAllCommunities();
      getUserCommunities();
    }
  }, [JwtToken]);

  // Update other communities whenever user or all communities change
  useEffect(() => {
    updateOtherCommunities();
  }, [allCommunities, leaderCommunities, memberCommunities]);

  const chatMessages = [
    {
      sender: "bot",
      text: "Hi there! I'm your WeFit assistant. How can I help you today?",
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    },
  ];

  const handleNavigateToHistory = () => {
    navigate("/history");
  };

  const handleToggleChatbot = () => {
    setIsChatbotOpen(!isChatbotOpen);
  };

  // Leaderboard Widget Component
  const LeaderboardWidget = () => {
    // Static leaderboard data for demonstration
    const leaderboardData = [
      { rank: 1, name: "Alex Johnson", score: 2450, avatar: "A" },
      { rank: 2, name: "Maya Chen", score: 2320, avatar: "M" },
      { rank: 3, name: "Tyler Smith", score: 2105, avatar: "T" },
      { rank: 4, name: "Sarah Adams", score: 1950, avatar: "S" },
      { rank: 5, name: "Ryan Park", score: 1820, avatar: "R" }
    ];

    return (
      <div className="p-8 rounded-2xl bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border border-[#3A225D] mb-8 shadow-lg">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-white text-2xl font-semibold tracking-tight flex items-center">
            <FaCrown className="text-yellow-400 mr-3" />
            Top Contributors
            <span className="text-purple-300 ml-2">({leaderboardData.length})</span>
          </h2>
        </div>

        <div className="bg-[#2B1748] border border-[#3A225D] rounded-xl overflow-hidden">
          {/* Leaderboard header */}
          <div className="flex items-center p-4 bg-[#3A225D] text-gray-300 text-sm font-medium">
            <div className="w-14 text-center">Rank</div>
            <div className="flex-1 ml-2">Member</div>
            <div className="w-24 text-right">Score</div>
          </div>

          {/* Leaderboard entries */}
          {leaderboardData.map((entry) => (
            <div
              key={entry.rank}
              className={`flex items-center p-4 border-b border-[#3A225D] last:border-b-0 hover:bg-[#341F52] transition-colors ${entry.rank === 1 ? 'bg-[#3A225D]/30' : ''
                }`}
            >
              {/* Rank */}
              <div className="w-14 text-center">
                {entry.rank <= 3 ? (
                  <div className={`
                    inline-flex items-center justify-center w-8 h-8 rounded-full
                    ${entry.rank === 1 ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
                      entry.rank === 2 ? 'bg-gradient-to-r from-gray-400 to-gray-300' :
                        'bg-gradient-to-r from-amber-700 to-amber-600'}
                  `}>
                    <span className="text-white font-bold">{entry.rank}</span>
                  </div>
                ) : (
                  <span className="text-gray-400 font-medium">{entry.rank}</span>
                )}
              </div>

              {/* User info */}
              <div className="flex-1 flex items-center">
                <div className="w-10 h-10 rounded-full bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] flex items-center justify-center text-white font-bold mr-3">
                  {entry.avatar}
                </div>
                <span className="text-white font-medium">{entry.name}</span>
              </div>

              {/* Score */}
              <div className="w-24 text-right">
                <span className="text-purple-300 font-bold">{entry.score}</span>
                <span className="text-gray-400 text-sm ml-1">pts</span>
              </div>
            </div>
          ))}
        </div>

        <div className="flex justify-center mt-4">
          <button className="px-4 py-2 text-sm text-purple-300 hover:text-white transition-colors">
            View Full Leaderboard
          </button>
        </div>
      </div>
    );
  };

  const CommunityCard = ({ community, isLeader, showJoinButton = false }) => {
    return (
      <div
        className={`relative flex flex-col p-6 rounded-xl border-2 border-[#3A225D] bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] hover:border-[#6B44AE] transition-all duration-300 shadow-lg hover:shadow-purple-500/20 ${showJoinButton ? '' : 'cursor-pointer group'}`}
        onClick={() => !showJoinButton && navigateToCommunity(community._id)}
      >
        {/* Glow effect on hover */}
        <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 bg-gradient-to-br from-purple-500/10 to-transparent transition-opacity duration-300 pointer-events-none"></div>

        <div className="flex justify-between items-start mb-3">
          <h3 className="text-white text-xl font-semibold tracking-tight">
            {community.name}
          </h3>
          {isLeader && (
            <div className="flex items-center bg-gradient-to-r from-yellow-500 to-yellow-600 px-3 py-1 rounded-full shadow-md">
              <FaCrown className="text-white mr-1 text-sm" />
              <span className="text-white text-xs font-medium">Leader</span>
            </div>
          )}
        </div>

        <p className="text-gray-300 text-sm mb-4 line-clamp-2 leading-relaxed">
          {community.description}
        </p>

        <div className="flex flex-wrap gap-2 mb-4">
          {community.tags?.map((tag, index) => (
            <span
              key={index}
              className="text-[#CDCDCD] text-xs bg-[#3A225D] px-3 py-1 rounded-full border border-[#4A2D7A]"
            >
              #{tag}
            </span>
          ))}
        </div>

        <div className="mt-auto flex items-center justify-between pt-4 border-t border-[#3A225D]">
          <div className="flex items-center bg-[#3A225D] px-3 py-1 rounded-full border border-[#4A2D7A]">
            <FaUsers className="text-purple-300 mr-2 text-sm" />
            <span className="text-white text-xs font-medium">
              {community.members?.length || 0} Members
            </span>
          </div>

          {showJoinButton ? (
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleJoinCommunity(community._id);
              }}
              className="px-4 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all shadow-md text-sm font-medium"
            >
              Join Community
            </button>
          ) : (
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] group-hover:from-[#7B54BE] group-hover:to-[#9A6FE3] transition-all">
              <span className="text-white font-bold text-sm">→</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Enhanced CommunitySection component
  const CommunitySection = ({ title, communities, isLeader, showJoinButton = false }) => {
    return (
      <div className="p-8 rounded-2xl bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border border-[#3A225D] mb-8 shadow-lg">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-white text-2xl font-semibold tracking-tight">
            {title} <span className="text-purple-300 ml-2">({communities.length})</span>
          </h2>

          {!showJoinButton && (
            <button
              onClick={() => isLeader ? setShowCreateModal(true) : setShowJoinModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all text-sm font-medium"
            >
              <FaPlus className="text-sm" />
              {isLeader ? "New Community" : "Join Community"}
            </button>
          )}
        </div>

        {communities.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {communities.map((community) => (
              <CommunityCard
                key={community._id}
                community={community}
                isLeader={isLeader}
                showJoinButton={showJoinButton}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-10">
            <div className="inline-block p-6 rounded-xl bg-[#2B1748] border border-[#3A225D]">
              <p className="text-gray-300 mb-4 text-sm">
                {isLeader ? "You haven't created any communities yet." :
                  showJoinButton ? "No communities available to join." : "You haven't joined any communities yet."}
              </p>
              {!showJoinButton && (
                <button
                  onClick={() => isLeader ? setShowCreateModal(true) : setShowJoinModal(true)}
                  className="px-6 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all text-sm font-medium"
                >
                  {isLeader ? "Create Your First Community" : "Browse Communities"}
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Enhanced CreateCommunityModal
  const CreateCommunityModal = () => {
    const [communityData, setCommunityData] = useState({
      name: "",
      description: "",
      tags: [],
      isPrivate: false,
      currentTag: ""
    });

    const handleChange = (e) => {
      const { name, value, type, checked } = e.target;
      setCommunityData(prev => ({
        ...prev,
        [name]: type === "checkbox" ? checked : value
      }));
    };

    const addTag = () => {
      if (communityData.currentTag.trim()) {
        setCommunityData(prev => ({
          ...prev,
          tags: [...prev.tags, prev.currentTag.trim()],
          currentTag: ""
        }));
      }
    };

    const removeTag = (indexToRemove) => {
      setCommunityData(prev => ({
        ...prev,
        tags: prev.tags.filter((_, index) => index !== indexToRemove)
      }));
    };

    const handleCreateCommunity = async () => {
      try {
        const dataToSubmit = {
          name: communityData.name,
          description: communityData.description,
          tags: communityData.tags,
        };

        await axios.post(
          `http://localhost:3000/community/create/${JwtToken}`,
          dataToSubmit,
          {
            headers: {
              Authorization: `Bearer ${JwtToken}`,
            },
          }
        );
        setShowCreateModal(false);
        navigate('/community/add-challenge');
        getUserCommunities();

      } catch (error) {
        console.log("Error creating community:", error);
        alert(error.response?.data?.message || "Failed to create community");
      }
    };

    return (
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border-2 border-[#6B44AE] rounded-2xl p-8 w-[500px] max-h-[90vh] overflow-y-auto shadow-xl">
          <h2 className="text-white text-2xl font-semibold mb-6 tracking-tight">Create Community</h2>

          <div className="space-y-5">
            <div>
              <label className="text-purple-300 text-sm font-medium block mb-2">Community Name*</label>
              <input
                type="text"
                name="name"
                value={communityData.name}
                onChange={handleChange}
                className="w-full bg-[#2B1748] text-white p-3 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all"
                placeholder="Enter community name"
                required
              />
            </div>

            <div>
              <label className="text-purple-300 text-sm font-medium block mb-2">Description*</label>
              <textarea
                name="description"
                value={communityData.description}
                onChange={handleChange}
                className="w-full bg-[#2B1748] text-white p-3 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all min-h-[120px]"
                placeholder="Describe your community"
                required
              />
            </div>

            <div>
              <label className="text-purple-300 text-sm font-medium block mb-2">Tags</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  name="currentTag"
                  value={communityData.currentTag}
                  onChange={handleChange}
                  onKeyDown={(e) => e.key === 'Enter' && addTag()}
                  className="flex-1 bg-[#2B1748] text-white p-3 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all"
                  placeholder="Add tags (press Enter or Add)"
                />
                <button
                  onClick={addTag}
                  className="px-4 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-lg hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all flex items-center justify-center"
                >
                  Add
                </button>
              </div>

              <div className="flex flex-wrap gap-2 mt-3">
                {communityData.tags.map((tag, index) => (
                  <div key={index} className="flex items-center gap-1 bg-[#3A225D] text-white px-3 py-1 rounded-full border border-[#4A2D7A]">
                    <span className="text-xs">#{tag}</span>
                    <button
                      onClick={() => removeTag(index)}
                      className="text-red-400 hover:text-red-300 text-xs"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="isPrivate"
                name="isPrivate"
                checked={communityData.isPrivate}
                onChange={handleChange}
                className="h-4 w-4 rounded border-[#3A225D] bg-[#2B1748] text-[#6B44AE] focus:ring-[#6B44AE]"
              />
              <label htmlFor="isPrivate" className="text-gray-300 ml-2 text-sm">Private Community</label>
            </div>
          </div>

          <div className="flex justify-end gap-3 mt-8">
            <button
              onClick={() => setShowCreateModal(false)}
              className="px-5 py-2 bg-[#3A225D] text-white rounded-lg hover:bg-[#4A2D7A] transition-all border border-[#4A2D7A] text-sm font-medium"
            >
              Cancel
            </button>
            <button
              onClick={handleCreateCommunity}
              disabled={!communityData.name || !communityData.description}
              className={`px-5 py-2 text-white rounded-lg text-sm font-medium ${!communityData.name || !communityData.description
                ? "bg-gray-600 cursor-not-allowed"
                : "bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all"
                }`}
            >
              Create Community
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Enhanced JoinCommunityModal
  const JoinCommunityModal = () => {
    const [availableCommunities, setAvailableCommunities] = useState([]);

    useEffect(() => {
      const userCommunityIds = [...leaderCommunities, ...memberCommunities].map(c => c._id);
      const available = allCommunities.filter(
        c => !userCommunityIds.includes(c._id)
      );
      setAvailableCommunities(available);
    }, [allCommunities, leaderCommunities, memberCommunities]);

    return (
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border-2 border-[#6B44AE] rounded-2xl p-8 w-[450px] max-h-[80vh] overflow-hidden">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-white text-2xl font-semibold tracking-tight">Join Community</h2>
            <button
              onClick={() => setShowJoinModal(false)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>

          <div className="relative mb-6">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <FaSearch className="text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Search communities..."
              className="w-full bg-[#2B1748] text-white pl-10 pr-4 py-2 rounded-lg border border-[#3A225D] focus:border-[#6B44AE] focus:ring-1 focus:ring-[#6B44AE] outline-none transition-all"
            />
          </div>

          {availableCommunities.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-gray-400 mb-4">No communities available to join</p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="px-4 py-2 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all text-sm font-medium"
              >
                Create New Community
              </button>
            </div>
          ) : (
            <div className="space-y-3 max-h-[50vh] overflow-y-auto pr-2">
              {availableCommunities.map((community) => (
                <div
                  key={community._id}
                  className="flex justify-between items-center p-4 rounded-xl bg-[#2B1748] border border-[#3A225D] hover:bg-[#3A225D] transition-colors cursor-pointer"
                  onClick={() => handleJoinCommunity(community._id)}
                >
                  <div>
                    <h3 className="text-white font-medium">{community.name}</h3>
                    {community.tags?.length > 0 && (
                      <p className="text-gray-400 text-sm mt-1">
                        {community.tags.map(tag => `#${tag}`).join(' ')}
                      </p>
                    )}
                  </div>
                  <button className="px-3 py-1 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full text-xs font-medium hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all">
                    Join
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#0F0A1A]">
      {/* Header remains the same */}
      <header className="desktop-home__header bg-[#1A0F2B] border-b border-[#3A225D]">
        <div className="desktop-home__logo-container">
          <img
            src="https://cdn.builder.io/api/v1/image/assets/TEMP/69e8365158abd202fc7d010edd0471beda6cd6aa?placeholderIfAbsent=true&apiKey=1455cb398c424e78afe4261a4bb08b71"
            alt="Logo"
            className="desktop-home__logo-image"
          />
          <div className="desktop-home__logo-text">WeFit</div>
        </div>
        <div className="flex gap-4 items-center">
          <div className="overflow-hidden">
            <ConnectWallet />
          </div>
          <button onClick={handleNavigateToHistory} className="rounded-full">
            <Avatar className="h-[63px] p-3.5 w-[63px] border-4 border-[#512E8B] rounded-full bg-[#413359] cursor-pointer hover:opacity-80 transition-opacity">
              <FaHistory color="white" size={30} />
            </Avatar>
          </button>
        </div>
      </header>

      {/* Main content with updated layout */}
      <main className="p-6 bg-gradient-to-br from-[#0F0A1A] to-[#1A0F2B] min-h-[calc(100vh-80px)]">
        {/* Enhanced top bar */}
        <div className="flex h-16 p-1 justify-between items-center rounded-full bg-[#2B1748] border border-[#3A225D] shadow-lg mb-8">
          <div className="flex items-center gap-3 ml-4">
            <button
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[#6B44AE] to-[#8A5FD3] text-white rounded-full hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all shadow-md text-sm font-medium"
              onClick={() => setShowCreateModal(true)}
            >
              <FaPlus className="text-sm" />
              Create Community
            </button>
          </div>
          <div className="text-white font-medium text-lg bg-[#3A225D] px-6 py-2 rounded-full border border-[#4A2D7A]">
            My Communities ({leaderCommunities.length + memberCommunities.length})
          </div>
          <div className="w-10"></div> {/* Spacer for balance */}
        </div>

        {/* Grid layout for main content and leaderboard */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main content - 2/3 width on large screens */}
          <div className="lg:col-span-2 space-y-8">
            <CommunitySection
              title="Your Communities"
              communities={leaderCommunities}
              isLeader={true}
            />

            <CommunitySection
              title="Joined Communities"
              communities={memberCommunities}
              isLeader={false}
            />

            <CommunitySection
              title="Discover Communities"
              communities={otherCommunities}
              isLeader={false}
              showJoinButton={true}
            />
          </div>

          {/* Sidebar with leaderboard - 1/3 width on large screens */}
          <div className="space-y-8">
            <LeaderboardWidget />

            {/* Optional: Additional widget could go here */}
            <div className="p-6 rounded-2xl bg-gradient-to-br from-[#1A0F2B] to-[#2B1748] border border-[#3A225D] shadow-lg">
              <div className="flex items-center mb-4">
                <FaTrophy className="text-yellow-400 mr-2" />
                <h3 className="text-white text-lg font-semibold tracking-tight">Your Achievements</h3>
              </div>
              <div className="text-center py-8 text-gray-400">
                <p>Join a community challenge to start earning achievements!</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {showCreateModal && <CreateCommunityModal />}
      {showJoinModal && <JoinCommunityModal />}

      {/* Floating elements */}
      <div className="fixed bottom-8 right-8 flex flex-col gap-4">
        <div
          className="chatbot-bubble w-14 h-14 bg-gradient-to-br from-[#6B44AE] to-[#8A5FD3] hover:from-[#7B54BE] hover:to-[#9A6FE3] transition-all shadow-lg"
          onClick={handleToggleChatbot}
        >
          <img src={chatbot} alt="Chatbot" className="p-3" />
        </div>
      </div>

      <DesktopChatbot
        isOpen={isChatbotOpen}
        onClose={() => setIsChatbotOpen(false)}
        messages={chatMessages}
      />
    </div>
  );
};

export default Community;