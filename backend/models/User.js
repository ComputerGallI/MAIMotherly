const mongoose = require('mongoose')

const quizResultSchema = new mongoose.Schema({
  quizType: {
    type: String,
    enum: ['personality', 'stress', 'love', '21questions'],
    required: true
  },
  result: String,
  score: Number,
  answers: [String],
  completedAt: { type: Date, default: Date.now }
})

const healthInfoSchema = new mongoose.Schema({
  age: Number,
  height: String,
  weight: String,
  conditions: [String],
  medications: [String],
  allergies: [String],
  lastPhysical: Date,
  lastDental: Date,
  insuranceProvider: String,
  insurancePlan: String,
  primaryCarePhysician: String
})

const subscriptionSchema = new mongoose.Schema({
  tier: {
    type: String,
    enum: ['free', 'basic', 'premium', 'ultimate'],
    default: 'free'
  },
  startDate: Date,
  endDate: Date,
  autoRenew: { type: Boolean, default: false },
  paymentMethod: String
})

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: String,
  password: { type: String, required: true },
  
  // quiz results stored in database
  quizResults: [quizResultSchema],
  
  // subscription and premium features
  subscription: subscriptionSchema,
  
  // health information for premium users
  healthInfo: healthInfoSchema,
  
  // google calendar integration
  googleCalendarAuth: {
    accessToken: String,
    refreshToken: String,
    isConnected: { type: Boolean, default: false }
  },
  
  // chat history and preferences
  chatPreferences: {
    preferredTone: { type: String, default: 'caring' },
    responseLength: { type: String, default: 'medium' },
    topicsToAvoid: [String]
  },
  
  // wellness tracking
  wellnessReminders: [{
    title: String,
    description: String,
    frequency: String,
    nextDue: Date,
    isActive: { type: Boolean, default: true }
  }],
  
  createdAt: { type: Date, default: Date.now },
  lastActive: { type: Date, default: Date.now }
})

// methods to get quiz summaries for AI
userSchema.methods.getQuizSummary = function() {
  const latest = {}
  
  this.quizResults.forEach(result => {
    if (!latest[result.quizType] || result.completedAt > latest[result.quizType].completedAt) {
      latest[result.quizType] = result
    }
  })
  
  return {
    personality: latest.personality?.result || '',
    stress: latest.stress?.result || '',
    love: latest.love?.result || '',
    questions21: latest['21questions']?.result || ''
  }
}

// check if user has premium features
userSchema.methods.hasPremiumAccess = function() {
  if (!this.subscription || this.subscription.tier === 'free') return false
  if (!this.subscription.endDate) return false
  return new Date() < this.subscription.endDate
}

// get user's current subscription level
userSchema.methods.getSubscriptionLevel = function() {
  if (!this.hasPremiumAccess()) return 'free'
  return this.subscription.tier
}

module.exports = mongoose.model('User', userSchema)