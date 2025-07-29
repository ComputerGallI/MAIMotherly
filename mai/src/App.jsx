import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <header>
    <h1>MAI – Your Motherly AI</h1>
    <p className="subtitle">I’m here to listen, guide, and care.</p>
    <nav>
      <button onclick="location.href='login.html'">Log In</button>
      <button onclick="location.href='chat.html'">Talk to MAI</button>
      <button onclick="location.href='quizzes.html'">Take a Quiz</button>
      <button onclick="location.href='profile.html'">My Profile</button>
    </nav>
  </header>

  <main className="home-banner">
    <img src='/images/mother-daughter-hug-serious_credit-shutterstock.jpg' alt="Motherly Comfort" className="hero-img"/>
    <h2 className="greeting">Welcome to your emotional wellness sanctuary.</h2>
  </main>

  <footer>
    <p>&copy; 2025 MAI by India Ratliff & Ryan Edwards</p>
  </footer>
    </>
  )
}

export default App
