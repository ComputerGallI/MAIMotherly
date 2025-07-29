const mongoose = require('mongoose')
const quizSchema = new mongoose.Schema({
  userId:String,
  type:String,
  answers:Object,
  summary:String,
  createdAt:Date
})
module.exports = mongoose.model('Quiz',quizSchema)
