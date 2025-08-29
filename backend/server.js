require('dotenv').config()
const express = require('express')
const mongoose = require('mongoose')
const session = require('express-session')
const MongoStore = require('connect-mongo')
const cors = require('cors')
const path = require('path')

// Import routes
const userRoutes = require('./routes/userRoutes')
const chatRoutes = require('./routes/chatRoutes')
const calendarRoutes = require('./routes/calendarRoutes')
const analyticsRoutes = require('./routes/analyticsRoutes')
const reminderRoutes = require('./routes/reminderRoutes')

const app = express()
const PORT = process.env.PORT || 5000

// Middleware
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:5500', 'http://localhost:5500'],
  credentials: true
}))
app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(express.static(path.join(__dirname, '../frontend')))

// Session configuration for Google OAuth
app.use(session({
  secret: process.env.JWT_SECRET || 'mai-secret-key-change-this',
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({
    mongoUrl: process.env.MONGO_URI
  }),
  cookie: {
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    maxAge: 1000 * 60 * 60 * 24 * 7 // 7 days
  }
}))

// Database connection
mongoose.connect(process.env.MONGO_URI || process.env.MONGODB_URI, {
  dbName: process.env.DB_NAME || 'maimotherly',
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('📁 Database connected'))
.catch(err => console.error('❌ Database connection failed:', err))

// Routes
app.use('/api/users', userRoutes)
app.use('/api/chat', chatRoutes)
app.use('/api/calendar', calendarRoutes)
app.use('/api/analytics', analyticsRoutes)
app.use('/api/reminders', reminderRoutes)

// Serve frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'))
})

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'backend running',
    database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
    timestamp: new Date().toISOString()
  })
})

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err.stack)
  res.status(500).json({
    success: false,
    error: process.env.NODE_ENV === 'development' ? err.message : 'Internal server error'
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' })
})

// Start server
app.listen(PORT, () => {
  console.log(`🚀 MAI server running on port ${PORT}`)
  console.log(`Frontend: http://localhost:${PORT}`)
  console.log(`API: http://localhost:${PORT}/api`)
  console.log(`💡 Make sure FastAPI is running on port 8000 for AI processing`)
  
  if (process.env.NODE_ENV === 'development') {
    console.log('Development mode - detailed errors enabled')
  }
})

module.exports = app