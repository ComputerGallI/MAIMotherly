// server.js
const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
require('dotenv').config()

const app = express()
app.use(cors())
app.use(express.json())

// connect db
mongoose.connect(process.env.MONGO_URI)
.then(()=>console.log('db connected'))
.catch(err=>console.error(err))

// routes
app.use('/api/users',require('./routes/userRoutes'))
app.use('/api/quiz',require('./routes/quizRoutes'))
app.use('/api/chat',require('./routes/chatRoutes'))

// start server
const PORT = process.env.PORT || 5000
app.listen(PORT,()=>console.log('server running on',PORT))
