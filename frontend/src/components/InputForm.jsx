import React, { useState } from "react";

export default function InputForm({ onAnalyze }) {
  const [text, setText] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    if (text.trim() === "") {
      setError("Please enter a sentence to analyze.");
      return;
    }

    setError("");
    onAnalyze(text);
    setText("");
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl mx-auto my-8">
      <textarea
        className={`w-full p-4 rounded-xl shadow-md text-gray-700 focus:outline-none focus:ring-2 transition-all
          ${error
            ? "border-2 border-red-400 focus:ring-red-400"
            : "focus:ring-purple-400"
          }`}
        placeholder="Type your sentence here..."
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          if (error) setError("");
        }}
        rows={3}
      />

      {/* Validation message */}
      {error && (
        <p className="mt-2 text-sm text-red-500 font-medium">
          {error}
        </p>
      )}

      <button
        type="submit"
        className="mt-4 w-full bg-green-500 text-white py-3 rounded-xl font-bold hover:bg-green-600 transition-all"
      >
        Analyze
      </button>
    </form>
  );
}
