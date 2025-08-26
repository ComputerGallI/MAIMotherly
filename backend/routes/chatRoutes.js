// Backend Chat Routes - Connects Frontend to AI
// This file takes chat messages from the frontend and sends them to the AI

const express = require('express')  // Web framework for Node.js
const axios = require('axios')      // Makes HTTP requests to the AI
const User = require('../models/User')  // Database model for users
const r = express.Router()          // Create a router for chat endpoints

// Main chat endpoint - this is where the magic happens!
r.post('/', async (req, res) => {
  try {
    console.log('🔥 New chat message received:', req.body)
    
    // Get the data sent from the frontend
    const { user_input, username } = req.body
    
    // Make sure they actually typed something
    if (!user_input || !user_input.trim()) {
      console.log('❌ ERROR: User sent empty message')
      return res.status(400).json({ 
        error: "Please enter a message to chat with MAI" 
      })
    }

    // Make sure we know who is talking
    if (!username || !username.trim()) {
      console.log('❌ ERROR: No username provided')
      return res.status(400).json({ 
        error: "You need to be logged in to chat with MAI" 
      })
    }

    console.log('📤 Sending to AI:', {
      message: user_input.trim(),
      user: username.trim(),
      subscription: "free"
    })

    // Send the message to our AI system (FastAPI)
    // The AI will automatically get the user's quiz results from MongoDB
    const fastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const fastApiResponse = await axios.post(`${fastApiUrl}/generate`, {
      user_input: user_input.trim(),    // What the user said
      username: username.trim(),        // Who said it (AI gets their quiz results automatically)
      subscription_tier: "free"         // Free or paid user
    }, {
      timeout: 30000,  // Wait up to 30 seconds for AI to respond (AI thinking takes time!)
      headers: {
        'Content-Type': 'application/json'
      }
    })

    console.log('✅ AI responded successfully:', {
      response_length: fastApiResponse.data?.response?.length || 0,
      suggestions_count: fastApiResponse.data?.suggestions?.length || 0
    })

    // Get the AI's response
    const aiResponse = fastApiResponse.data?.response
    const suggestions = fastApiResponse.data?.suggestions || []

    // Make sure the AI actually gave us a response
    if (!aiResponse) {
      throw new Error('AI did not provide a response')
    }

    // Send the AI's response back to the frontend
    res.json({ 
      response: aiResponse,           // What MAI said
      suggestions: suggestions,       // Helpful suggestions for the user
      ai_powered: true,              // Confirm this came from real AI
      no_templates: true,            // Confirm no boring templates were used
      debug_info: {
        fastapi_called: true,
        response_length: aiResponse.length,
        suggestions_count: suggestions.length,
        username: username
      }
    })

  } catch (error) {
    console.error('💥 Chat error:', error.message)
    
    // Figure out what went wrong and give helpful error messages
    let errorMessage = "I'm having trouble connecting to my AI brain right now."
    let statusCode = 500
    
    // Different types of errors need different messages
    if (error.code === 'ECONNREFUSED') {
      // The AI server isn't running
      errorMessage = "My AI brain is offline. Please start the FastAPI server (python main.py)."
      statusCode = 503
      console.log('🚨 ERROR: FastAPI server is not running!')
    } else if (error.code === 'ETIMEDOUT') {
      // The AI is taking too long to think
      errorMessage = "I'm thinking extra hard about your question - it's taking longer than usual. Please try again."
      statusCode = 504
      console.log('⏰ ERROR: AI took too long to respond')
    } else if (error.response?.status === 503) {
      // The AI system isn't ready (missing Gemini key or training data)
      errorMessage = "My AI system isn't fully set up yet. Please check the AI configuration."
      statusCode = 503
      console.log('⚙️ ERROR: AI system not ready:', error.response.data)
    } else if (error.response?.status === 500) {
      // The AI had an internal error
      errorMessage = "My AI brain encountered an error while thinking. Please try again."
      statusCode = 500
      console.log('🤖 ERROR: AI system error:', error.response.data)
    } else if (error.response) {
      // Some other HTTP error from the AI
      errorMessage = `My AI service had an error (code ${error.response.status}). Please try again.`
      statusCode = error.response.status
      console.log('🌐 ERROR: FastAPI returned error:', error.response.status, error.response.data)
    }
    
    // Send the error back to the frontend
    res.status(statusCode).json({ 
      error: errorMessage,
      ai_error: true,
      debug_info: {
        error_type: error.code || 'unknown',
        fastapi_url: process.env.FASTAPI_URL || 'http://127.0.0.1:8000',
        status_code: error.response?.status || 'no_response',
        help: "Make sure FastAPI is running: cd fastapi_ai && python main.py"
      }
    })
  }
})

// Get a user's chat history and statistics
r.get('/history/:username', async (req, res) => {
  try {
    const { username } = req.params
    
    if (!username) {
      return res.status(400).json({ error: 'Username is required' })
    }

    console.log(`📊 Getting chat stats for ${username}`)

    // Ask the AI system for user statistics
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

// Test endpoint to check if the AI is working
r.get('/test', async (req, res) => {
  try {
    const fastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const response = await axios.get(`${fastApiUrl}/health`)
    
    res.json({
      backend_status: "OK",
      fastapi_status: response.data,
      connection: "SUCCESS"
    })
    
  } catch (error) {
    res.status(503).json({
      backend_status: "OK", 
      fastapi_status: "FAILED",
      connection: "FAILED",
      error: error.message,
      help: "Start FastAPI with: cd fastapi_ai && python main.py"
    })
  }
})

// Export the router so other files can use it
module.exports = r