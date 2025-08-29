const express = require('express')
const ChatLog = require('../models/ChatLog')
const User = require('../models/User')
const r = express.Router()

/**
 * Get user's chat analytics
 * Premium users get detailed analytics
 */
r.get('/chat/:username', async (req, res) => {
  try {
    const username = req.params.username
    const user = await User.findOne({ username })
    
    // Basic stats for all users
    const chatStats = await ChatLog.aggregate([
      { $match: { username } },
      { $group: {
        _id: null,
        totalChats: { $sum: 1 },
        avgMessageLength: { $avg: '$messageLength' },
        totalChatsToday: {
          $sum: {
            $cond: [
              { $gte: ['$createdAt', new Date(new Date().setHours(0,0,0,0))] },
              1,
              0
            ]
          }
        }
      }}
    ])

    const stats = chatStats[0] || { totalChats: 0, avgMessageLength: 0, totalChatsToday: 0 }
    
    // Add premium analytics for paid users
    if (user && user.hasPremiumAccess()) {
      const topTopics = await ChatLog.aggregate([
        { $match: { username } },
        { $unwind: '$detectedTopics' },
        { $group: { _id: '$detectedTopics', count: { $sum: 1 } }},
        { $sort: { count: -1 } },
        { $limit: 5 }
      ])
      
      stats.topTopics = topTopics
      stats.subscriptionTier = user.getSubscriptionLevel()
    }

    res.json(stats)
  } catch (error) {
    console.error('Error getting chat analytics:', error)
    res.status(500).json({ error: 'Could not retrieve analytics' })
  }
})

module.exports = r
