import React, { useState } from "react";
import InputForm from "./components/InputForm";
import SentimentCard from "./components/SentimentCard";
import RecentAnalyses from "./components/RecentAnalyses";
import { analyzeSentiment } from "./api";

export default function App() {
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const handleAnalyze = async (text) => {
    const res = await analyzeSentiment(text);
    if (res) {
      setResult(res);
      setHistory([res, ...history].slice(0, 10));
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-100 to-purple-200 font-sans">
      <header className="text-center py-6 font-bold text-3xl text-gray-800">
        Sentiment Analysis 🚀
      </header>
      <InputForm onAnalyze={handleAnalyze} />
      <SentimentCard result={result} />
      <RecentAnalyses history={history} />
      <footer className="text-center mt-auto py-4 text-gray-500 text-sm">
        © 2025 Sentiment Analyzer | Powered by FastAPI
      </footer>
    </div>
  );
}
