import React from "react";

const emojiMap = {
  positive: "😀",
  neutral: "😐",
  negative: "😔",
};

export default function RecentAnalyses({ history }) {
  if (!history.length) return null;

  return (
    <div className="max-w-xl mx-auto my-6">
      <h2 className="text-gray-700 font-semibold mb-3">Recent Analyses</h2>
      <div className="space-y-3">
        {history.map((item, idx) => (
          <div key={idx} className="p-4 rounded-xl bg-white shadow-sm flex justify-between items-center">
            <div>
              <span className="mr-2">{emojiMap[item.sentiment]}</span>
              {item.text}
            </div>
            <div className="text-gray-600 font-medium">{(item.confidence * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}
