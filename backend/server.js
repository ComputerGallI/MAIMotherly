//server start
const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
require('dotenv').config()

const app = express()
app.use(cors())
app.use(express.json())  // search through json strings

//db connect
mongoose.connect(process.env.MONGO_URI)
.then(()=>console.log('db ok'))
.catch(err=>console.error('db fail',err))

// routes
app.use('/api/users',require('./routes/userRoutes'))
app.use('/api/quiz',require('./routes/quizRoutes'))
app.use('/api/chat',require('./routes/chatRoutes'))

//listen
const PORT = process.env.PORT || 5000
app.listen(PORT,()=>console.log('server on',PORT))
