const express = require('express')
const cors = require('cors')
const session = require('express-session')
const dotenv = require('dotenv')
const connect = require('./config/db')

dotenv.config()

const app = express()

// basic middleware stuff
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:5500', 'http://localhost:8080'],
  credentials: true // needed for sessions
}))
app.use(express.json())

// sessions for google oauth
app.use(session({
  secret: process.env.JWT_SECRET || 'mai-secret-key-change-this',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: false } // set to true if using https
}))

// connect to mongodb
connect()

// basic route
app.get('/', (req, res) => res.send('mai backend running'))

// hook up all our routes
app.use('/api/users', require('./routes/userRoutes'))
app.use('/api/chat', require('./routes/chatRoutes'))
app.use('/api/reminders', require('./routes/reminderRoutes'))
app.use('/api/calendar', require('./routes/calendarRoutes'))

const PORT = process.env.PORT || 5000
app.listen(PORT, () => console.log(`server running on port ${PORT}`))