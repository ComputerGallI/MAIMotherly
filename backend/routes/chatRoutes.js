// DEMO VERSION - Hardcoded responses for demo users
// This ensures your demo works perfectly with the exact responses you need
// Replace backend/routes/chatRoutes.js with this file for the demo

const express = require('express')
const axios = require('axios')
const User = require('../models/User')
const r = express.Router()

// Demo responses for specific users and prompts
const DEMO_RESPONSES = {
  "IndieMovies": {
    // Conversation tracking for multi-turn conversations
    conversation_count: 0,
    responses: [
      {
        // Prompt 1
        keywords: ["hello", "trouble sleeping", "new job", "sleep"],
        response: "Starting a new job can turn your whole rhythm upside down. Trouble sleeping is a common response when your body and mind are adjusting. Given your thoughtful, introspective nature as an INFP, you probably process work stress deeply. Let's create some gentle cues to help you wind down.",
        suggestions: [
          "Shut off screens 30 minutes before bed",
          "Write down lingering thoughts in a journal before sleep", 
          "Try a breathing exercise: inhale 4, exhale 6",
          "Keep lights dim an hour before bedtime"
        ]
      },
      {
        // Prompt 2  
        keywords: ["thank you", "imposter syndrome", "imposter", "don't belong"],
        response: "Feeling like you don't belong, even when you're qualified, is something many people quietly struggle with. Your feelings are valid. Since words of affirmation are meaningful to you, let's ground your worth in small reminders of what you've already achieved and kind words you can tell yourself.",
        suggestions: [
          "List 3 ways you contributed at work this week",
          "Talk through feelings with a supportive friend",
          "Keep a small note with encouraging words near your desk",
          "Practice self-compassion: say one kind thing to yourself tonight"
        ]
      },
      {
        // Prompt 3
        keywords: ["get off so late", "don't have time", "exercise", "cook", "fix this"],
        response: "It sounds like you're juggling long hours and self-care. With your moderate stress levels and preference for meaningful routines, instead of big tasks, we can shrink them into bite-sized steps that still give you energy and comfort.",
        suggestions: [
          "Prep quick healthy snacks on weekends (fruit, nuts, wraps)",
          "Try short body-weight stretches before bed instead of long workouts", 
          "Batch cook one meal on a day off and reheat through the week",
          "Choose one relaxing ritual (tea, stretching, journal) before sleep"
        ]
      }
    ]
  },
  "Ryanator": {
    conversation_count: 0,
    responses: [
      {
        // Prompt 1
        keywords: ["feeling fired up", "energy", "crash", "surge"],
        response: "That surge and drop of energy can feel intense. With your ENFP personality, you naturally have bursts of enthusiasm that can sometimes lead to burnout. It might help to pace your energy gently, like smoothing out a wave, so the crash doesn't feel as sharp.",
        suggestions: [
          "Take a 5-minute break before pushing too far",
          "Eat a balanced snack to stabilize energy",
          "Jot down what triggered the energy spike", 
          "Try a calming breath before sleep"
        ]
      },
      {
        // Prompt 2
        keywords: ["never thought", "hyperactive", "normal", "being hyperactive"],
        response: "High energy bursts can happen for many reasons, and it doesn't always mean something is wrong. Given your moderate stress levels and energetic nature, what matters is noticing patterns—how often it happens, and how it affects your daily life.",
        suggestions: [
          "Track when energy highs/lows occur",
          "Limit caffeine after noon",
          "Schedule small check-in breaks",
          "Ask a trusted person if they notice patterns too"
        ]
      },
      {
        // Prompt 3
        keywords: ["center myself", "help center", "considering that", "what can i do"],
        response: "You can bring balance by anchoring your body and calming your mind. Think of it like lowering the volume when things feel loud inside. Since physical touch is one of your love languages, grounding exercises that involve your body might feel especially centering.",
        suggestions: [
          "Do a slow body scan from head to toe",
          "Try box breathing (inhale 4, hold 4, exhale 4, hold 4)",
          "Stretch arms and back gently",
          "Step outside for fresh air and grounding"
        ]
      },
      {
        // Prompt 4
        keywords: ["knee", "hurts", "knee hurts", "pain"],
        response: "I hear you—knee pain can get in the way of centering exercises. Let's adapt with movements that are gentle on your joints. Since acts of service matter to you, think of gentle stretching as a kind act you're doing for your own body. A little stretching beforehand can also ease the strain.",
        suggestions: [
          "Before activity: straighten your leg, gently pull your foot toward you for 20 seconds",
          "Do quad stretch: stand, bend one knee, hold ankle, keep knees together",
          "Replace deep bends with seated stretches", 
          "Apply gentle heat pack before bedtime"
        ]
      }
    ]
  }
}

// Function to check if user input matches demo keywords
function findDemoResponse(username, userInput) {
  const demoUser = DEMO_RESPONSES[username]
  if (!demoUser) return null
  
  const userInputLower = userInput.toLowerCase()
  
  // Get the next response in sequence for this user
  const responseIndex = demoUser.conversation_count
  
  if (responseIndex >= demoUser.responses.length) {
    // If we've used all demo responses, return null to use AI
    return null
  }
  
  const targetResponse = demoUser.responses[responseIndex]
  
  // Check if the user input contains any of the expected keywords
  const hasKeywords = targetResponse.keywords.some(keyword => 
    userInputLower.includes(keyword.toLowerCase())
  )
  
  if (hasKeywords) {
    // Increment conversation count for this user
    demoUser.conversation_count++
    
    console.log(`🎭 DEMO: Using hardcoded response ${responseIndex + 1} for ${username}`)
    
    return {
      response: targetResponse.response,
      suggestions: targetResponse.suggestions
    }
  }
  
  return null
}

// Main chat endpoint with demo override
r.post('/', async (req, res) => {
  try {
    console.log('🎭 DEMO Chat request received:', req.body)
    
    const { user_input, username } = req.body
    
    if (!user_input || !user_input.trim()) {
      console.log('ERROR: Empty user input')
      return res.status(400).json({ 
        error: "Please enter a message to chat with MAI" 
      })
    }

    if (!username || !username.trim()) {
      console.log('ERROR: No username provided')
      return res.status(400).json({ 
        error: "You need to be logged in to chat with MAI" 
      })
    }

    console.log(`🎭 Checking demo responses for user: ${username}`)
    console.log(`🎭 User input: "${user_input}"`)

    // Check for demo responses first
    const demoResponse = findDemoResponse(username, user_input)
    
    if (demoResponse) {
      console.log('✅ DEMO: Using hardcoded response for demo')
      
      // Return the demo response
      res.json({ 
        response: demoResponse.response,
        suggestions: demoResponse.suggestions,
        demo_mode: true,
        ai_powered: true,
        no_templates: false,  // This IS technically a template for demo
        debug_info: {
          demo_response: true,
          username: username,
          response_length: demoResponse.response.length,
          suggestions_count: demoResponse.suggestions.length
        }
      })
      return
    }

    console.log('🤖 No demo match - using real AI system')

    // If no demo response found, use the real AI system
    const fastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const fastApiResponse = await axios.post(`${fastApiUrl}/generate`, {
      user_input: user_input.trim(),
      username: username.trim(),
      subscription_tier: "free"
    }, {
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    })

    console.log('✅ Real AI responded successfully')

    const aiResponse = fastApiResponse.data?.response
    const suggestions = fastApiResponse.data?.suggestions || []

    if (!aiResponse) {
      throw new Error('AI did not provide a response')
    }

    res.json({ 
      response: aiResponse,
      suggestions: suggestions,
      demo_mode: false,
      ai_powered: true,
      no_templates: true,
      debug_info: {
        fastapi_called: true,
        response_length: aiResponse.length,
        suggestions_count: suggestions.length,
        username: username
      }
    })

  } catch (error) {
    console.error('💥 Chat error:', error.message)
    
    let errorMessage = "I'm having trouble connecting to my AI brain right now."
    let statusCode = 500
    
    if (error.code === 'ECONNREFUSED') {
      errorMessage = "My AI brain is offline. Please start the FastAPI server (python main.py)."
      statusCode = 503
      console.log('🚨 ERROR: FastAPI server is not running!')
    } else if (error.code === 'ETIMEDOUT') {
      errorMessage = "I'm thinking extra hard about your question - it's taking longer than usual. Please try again."
      statusCode = 504
      console.log('⏰ ERROR: AI took too long to respond')
    } else if (error.response?.status === 503) {
      errorMessage = "My AI system isn't fully set up yet. Please check the AI configuration."
      statusCode = 503
      console.log('⚙️ ERROR: AI system not ready:', error.response.data)
    } else if (error.response?.status === 500) {
      errorMessage = "My AI brain encountered an error while thinking. Please try again."
      statusCode = 500
      console.log('🤖 ERROR: AI system error:', error.response.data)
    } else if (error.response) {
      errorMessage = `My AI service had an error (code ${error.response.status}). Please try again.`
      statusCode = error.response.status
      console.log('🌐 ERROR: FastAPI returned error:', error.response.status, error.response.data)
    }
    
    res.status(statusCode).json({ 
      error: errorMessage,
      ai_error: true,
      demo_mode: false,
      debug_info: {
        error_type: error.code || 'unknown',
        fastapi_url: process.env.FASTAPI_URL || 'http://127.0.0.1:8000',
        status_code: error.response?.status || 'no_response',
        help: "Make sure FastAPI is running: cd fastapi_ai && python main.py"
      }
    })
  }
})

// Reset demo conversation counters (useful for testing)
r.post('/reset-demo', (req, res) => {
  Object.keys(DEMO_RESPONSES).forEach(username => {
    DEMO_RESPONSES[username].conversation_count = 0
  })
  
  console.log('🎭 DEMO: Reset all conversation counters')
  res.json({ message: 'Demo conversation counters reset' })
})

// Check demo status
r.get('/demo-status', (req, res) => {
  const status = {}
  Object.keys(DEMO_RESPONSES).forEach(username => {
    status[username] = {
      conversation_count: DEMO_RESPONSES[username].conversation_count,
      total_responses: DEMO_RESPONSES[username].responses.length
    }
  })
  
  res.json({
    demo_mode: true,
    demo_users: Object.keys(DEMO_RESPONSES),
    status: status
  })
})

// Get user chat history (same as before)
r.get('/history/:username', async (req, res) => {
  try {
    const { username } = req.params
    
    if (!username) {
      return res.status(400).json({ error: 'Username is required' })
    }

    console.log(`📊 Getting chat stats for ${username}`)

    const fastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const response = await axios.get(`${fastApiUrl}/user-stats/${username}`)
    
    console.log(`✅ Retrieved stats for ${username}:`, response.data)
    res.json(response.data)
    
  } catch (error) {
    console.error('📊 Error getting chat history:', error.message)
    res.status(500).json({ 
      error: 'Could not retrieve chat history',
      details: error.message 
    })
  }
})

// Test endpoint
r.get('/test', async (req, res) => {
  try {
    const fastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const response = await axios.get(`${fastApiUrl}/health`)
    
    res.json({
      backend_status: "OK",
      demo_mode: true,
      demo_users: Object.keys(DEMO_RESPONSES),
      fastapi_status: response.data,
      connection: "SUCCESS"
    })
    
  } catch (error) {
    res.status(503).json({
      backend_status: "OK", 
      demo_mode: true,
      fastapi_status: "FAILED",
      connection: "FAILED",
      error: error.message,
      help: "Start FastAPI with: cd fastapi_ai && python main.py"
    })
  }
})

module.exports = r