import React from "react";
import ConfidenceBar from "./ConfidenceBar";

const emojiMap = {
  positive: "😀",
  neutral: "😐",
  negative: "😔",
};

const colorMap = {
  positive: "text-green-500",
  neutral: "text-yellow-500",
  negative: "text-red-500",
};

export default function SentimentCard({ result }) {
  if (!result) return null;

  return (
    <div className="max-w-xl mx-auto p-6 rounded-2xl bg-white shadow-lg my-6 animate-fadeIn">
      <p className="text-gray-800 text-lg mb-4">"{result.text}"</p>
      <div className="flex items-center space-x-3 mb-3">
        <span className={`text-4xl ${colorMap[result.sentiment]}`}>
          {emojiMap[result.sentiment]}
        </span>
        <span className={`text-2xl font-bold ${colorMap[result.sentiment]}`}>
          {result.sentiment.charAt(0).toUpperCase() + result.sentiment.slice(1)}
        </span>
      </div>
      <ConfidenceBar confidence={result.confidence} />
    </div>
  );
}
