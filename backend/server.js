const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
const session = require('express-session')
const MongoStore = require('connect-mongo')
require('dotenv').config()

const app = express()

// Middleware
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:5500', 'http://localhost:5500'],
  credentials: true
}))
app.use(express.json())
app.use(express.urlencoded({ extended: true }))

// Session configuration
app.use(session({
  secret: process.env.JWT_SECRET || 'mai-secret-key-change-this',
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({
    mongoUrl: process.env.MONGO_URI
  }),
  cookie: {
    secure: false, // set to true in production with HTTPS
    httpOnly: true,
    maxAge: 1000 * 60 * 60 * 24 * 7 // 7 days
  }
}))

// Database connection
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('database connected'))
  .catch(err => console.log('database connection failed:', err.message))

// Import routes
const userRoutes = require('./routes/userRoutes')
const chatRoutes = require('./routes/chatRoutes')
const calendarRoutes = require('./routes/calendarRoutes')

// Use routes
app.use('/api/users', userRoutes)
app.use('/api/chat', chatRoutes)
app.use('/api/calendar', calendarRoutes)

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'backend running',
    database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
    timestamp: new Date().toISOString()
  })
})

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Server error:', error)
  res.status(500).json({ error: 'Internal server error' })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' })
})

const PORT = process.env.PORT || 5000

app.listen(PORT, () => {
  console.log(`server running on port ${PORT}`)
})

module.exports = app