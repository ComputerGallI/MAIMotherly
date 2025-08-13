const mongoose = require('mongoose')

const chatSchema = new mongoose.Schema({
  username: String,
  input: String,
  response: String,
  retrieved: [String],
  createdAt: { type: Date, default: Date.now }
})

module.exports = mongoose.model('ChatLog', chatSchema)
