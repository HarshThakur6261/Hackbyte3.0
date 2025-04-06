import React from "react";

const Leaderboard = ({ onClick }) => {
  return (
    <div 
      className="w-[308px] p-5 rounded-[19px] h-fit bg-[#4a336e5c] cursor-pointer hover:bg-[#5a3f7e] transition-colors"
      onClick={onClick}
    >
      <div className="flex flex-col items-center gap-5">
        <h2 className="text-white text-xl font-medium">Leaderboard</h2>
        <img 
          src="https://cdn.builder.io/api/v1/image/assets/TEMP/5ee0513993b0ba538478b5f1d93cdfa9972dd1ed" 
          alt="Trophy" 
          className="rounded-full w-[99px] h-[99px] bg-gradient-to-br from-purple-600 to-red-400" 
        />
        <div className="w-full p-5 rounded-[11px] border-2 border-[#301F4C] bg-[#1A0F2B]">
          <div className="flex justify-between mb-3">
            <span className="text-white text-sm font-medium">
              Rank No.
            </span>
            <span className="text-[#ABABAB] text-sm font-medium">
              Out of ..
            </span>
          </div>
          <div className="flex flex-col gap-3">
            {[
              { rank: 1, bg: "#FFD700" },
              { rank: 2, bg: "#C0C0C0" },
              { rank: 3, bg: "#CD7F32" },
              { rank: 4, bg: "#2A9D90" },
            ].map((item) => (
              <div
                key={item.rank}
                className="flex items-center gap-2.5 pb-3 border-b border-[#7E7E7E]"
              >
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium"
                  style={{ backgroundColor: item.bg }}
                >
                  {item.rank}
                </div>
                <span className="text-white text-sm">
                  Community Name
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
export default Leaderboard;