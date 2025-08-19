const mongoose = require('mongoose')

const chatLogSchema = new mongoose.Schema({
  username: { type: String, required: true, index: true },
  userMessage: { type: String, required: true },
  aiResponse: { type: String, required: true },
  
  // context information
  quizContext: String, // personality, stress level, etc.
  subscriptionTier: { type: String, default: 'free' },
  
  // conversation metadata
  messageLength: Number,
  responseTime: Number, // in milliseconds
  userSatisfaction: Number, // 1-5 rating if user provides feedback
  
  // analysis fields for premium insights
  detectedMood: String,
  detectedTopics: [String],
  suggestionsOffered: [String],
  
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
})

// indexes for efficient queries
chatLogSchema.index({ username: 1, createdAt: -1 })
chatLogSchema.index({ detectedTopics: 1 })
chatLogSchema.index({ subscriptionTier: 1 })

// pre-save middleware to calculate message length
chatLogSchema.pre('save', function(next) {
  if (this.userMessage) {
    this.messageLength = this.userMessage.length
  }
  this.updatedAt = new Date()
  next()
})

// instance methods
chatLogSchema.methods.addFeedback = function(satisfaction, notes) {
  this.userSatisfaction = satisfaction
  if (notes) {
    this.feedbackNotes = notes
  }
  return this.save()
}

// static methods for analytics
chatLogSchema.statics.getUserStats = async function(username) {
  const stats = await this.aggregate([
    { $match: { username } },
    {
      $group: {
        _id: null,
        totalChats: { $sum: 1 },
        avgMessageLength: { $avg: '$messageLength' },
        totalTopics: { $addToSet: '$detectedTopics' },
        lastChat: { $max: '$createdAt' },
        firstChat: { $min: '$createdAt' }
      }
    }
  ])
  
  return stats[0] || {
    totalChats: 0,
    avgMessageLength: 0,
    totalTopics: [],
    lastChat: null,
    firstChat: null
  }
}

chatLogSchema.statics.getPopularTopics = async function(timeframe = 30) {
  const cutoffDate = new Date()
  cutoffDate.setDate(cutoffDate.getDate() - timeframe)
  
  const topics = await this.aggregate([
    { $match: { createdAt: { $gte: cutoffDate } } },
    { $unwind: '$detectedTopics' },
    { $group: { _id: '$detectedTopics', count: { $sum: 1 } } },
    { $sort: { count: -1 } },
    { $limit: 10 }
  ])
  
  return topics
}

module.exports = mongoose.model('ChatLog', chatLogSchema)