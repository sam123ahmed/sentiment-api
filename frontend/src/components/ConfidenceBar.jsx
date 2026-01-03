import React from "react";

export default function ConfidenceBar({ confidence }) {
  return (
    <div className="w-full bg-gray-200 rounded-full h-4">
      <div
        className="h-4 rounded-full transition-all"
        style={{
          width: `${(confidence * 100).toFixed(0)}%`,
          background: `linear-gradient(90deg, #4caf50, #f44336)`,
        }}
      />
      <p className="text-gray-600 text-sm mt-1">{(confidence * 100).toFixed(1)}%</p>
    </div>
  );
}
