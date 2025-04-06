const StatisticCard = ({ title, value, textColor }) => {
  return (
    <article className="flex flex-col gap-2.5 px-9 py-8 bg-gray-900 rounded-3xl border-2 border-gray-700 h-[151px] min-w-[318px]">
      <h2 className="text-lg text-stone-300">{title}</h2>
      <p className={`text-4xl ${textColor}`}>{value}</p>
    </article>
  );
};

export default StatisticCard;
