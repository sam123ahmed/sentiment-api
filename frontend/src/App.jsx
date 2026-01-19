
import { useState } from 'react'
import axios from 'axios'
import Lottie from 'lottie-react'
import thinking from './lottie.json'

const API = "http://18.234.231.208:8000/api/v1/predict"

const map = {
  positive: { emoji: "😊", color: "pos" },
  negative: { emoji: "😠", color: "neg" },
  neutral:  { emoji: "😐", color: "neu" }
}

export default function App() {
  const [text, setText] = useState("")
  const [res, setRes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fb, setFb] = useState(null)
  const [correct, setCorrect] = useState(null)

  const submit = async () => {
    if (!text.trim()) return
    setLoading(true)
    setRes(null)
    setFb(null)
    setCorrect(null)
    const r = await axios.post(API, { text })
    setRes(r.data)
    setLoading(false)
  }

  const confidence = res ? Math.round(res.confidence * 100) : 0
  const sentiment = res?.sentiment
  const ui = sentiment ? map[sentiment] : null

  return (
    <div className="page">
      <div className="shell">
        <h1>Sentiment Analyzer</h1>
        <p className="tagline">Understand emotion with AI</p>

        <textarea
          placeholder="Type a sentence and let AI analyze the emotion…"
          value={text}
          onChange={e => setText(e.target.value)}
        />

        <button onClick={submit} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>

        {loading && (
          <div className="thinking">
            <Lottie animationData={thinking} loop />
            <span>AI is thinking…</span>
          </div>
        )}

        {res && (
          <div className={`result ${ui.color}`}>
            <div className="emotion">
              <span className="emoji">{ui.emoji}</span>
              <span className="label">{sentiment}</span>
            </div>

            <div className="ring" style={{ '--p': confidence }}>
              <span>{confidence}%</span>
            </div>

            {!fb && (
              <div className="feedback">
                <p>Was this prediction accurate?</p>
                <div className="fb-buttons">
                  <button onClick={() => setFb("up")}>👍 Yes</button>
                  <button onClick={() => setFb("down")}>👎 No</button>
                </div>
              </div>
            )}

            {fb === "up" && (
              <div className="thanks">Thanks for your feedback! 🎉</div>
            )}

            {fb === "down" && !correct && (
              <div className="correct">
                <p>What should it be?</p>
                <div className="fb-buttons">
                  {["positive","neutral","negative"].map(s => (
                    <button key={s} onClick={() => setCorrect(s)}>{s}</button>
                  ))}
                </div>
              </div>
            )}

            {correct && (
              <div className="thanks">
                Feedback recorded as <b>{correct}</b> ✅
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
