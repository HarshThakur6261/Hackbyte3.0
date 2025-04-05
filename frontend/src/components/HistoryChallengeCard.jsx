import ProgressCircle from "./ui/ProgressCircleHistory";
const ChallengeCard = ({
    title,
    date,
    amount,
    status,
    progress,
    isSuccess,
    handlecompleteExercise,
    exercise,
  }) => {
    return (
      <article className="flex flex-col gap-5 px-9 py-8 bg-gray-900 rounded-3xl border-2 border-gray-700 w-[433px]">
        <div className="flex justify-between items-center">
          <div className="flex flex-col">
            <h3 className="text-2xl text-zinc-300">{title}</h3>
            <time className="text-xl text-zinc-300">{date}</time>
          </div>
          <div
            className={`px-4 py-1 text-xl ${isSuccess ? "bg-indigo-700" : "bg-red-600"} rounded-[100px] text-zinc-300`}
          >
            {amount}
          </div>
        </div>
        <div className="flex gap-6 items-center">
          <div className="h-[90px] w-[85px]" >
            <ProgressCircle handlecompleteExercise={progress} exercise={progress} />
          </div>
          <div className="flex flex-col gap-px">
            <p className="text-xl text-zinc-300">{status}</p>
            <p className="text-2xl text-zinc-300">{progress}</p>
          </div>
        </div>
      </article>
    );
  };
  
  export default ChallengeCard;
  