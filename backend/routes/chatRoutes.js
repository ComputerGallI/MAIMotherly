const express = require('express')
const axios = require('axios')
const r = express.Router()

// Chat endpoint that calls FastAPI for AI processing
r.post('/', async (req, res) => {
  try {
    console.log('Chat request received:', req.body)
    
    const { user_input, username, quiz_summary } = req.body
    
    if (!user_input || !user_input.trim()) {
      console.log('ERROR: Empty user input')
      return res.json({ 
        response: "I'm here to listen. What's on your mind?" 
      })
    }

    console.log('Calling FastAPI with:', {
      user_input: user_input.trim(),
      quiz_summary: quiz_summary || "",
      subscription_tier: "free"
    })

    // Call FastAPI service for AI processing
    const fastApiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const fastApiResponse = await axios.post(`${fastApiUrl}/generate`, {
      user_input: user_input.trim(),
      quiz_summary: quiz_summary || "",
      subscription_tier: "free"
    }, {
      timeout: 15000, // 15 second timeout
      headers: {
        'Content-Type': 'application/json'
      }
    })

    console.log('FastAPI response received:', fastApiResponse.data)

    const aiResponse = fastApiResponse.data?.response || "I want to help you with that. Can you tell me more?"
    const suggestions = fastApiResponse.data?.suggestions || []

    res.json({ 
      response: aiResponse,
      suggestions: suggestions,
      debug_info: {
        fastapi_called: true,
        response_length: aiResponse.length
      }
    })

  } catch (error) {
    console.error('Chat error:', error.message)
    
    // Provide specific error information for debugging
    let errorResponse = "I'm having trouble accessing my knowledge right now, but I'm still here to listen. "
    
    if (error.code === 'ECONNREFUSED') {
      errorResponse += "It seems my AI brain isn't responding. Can you try restarting the FastAPI server?"
      console.log('ERROR: FastAPI server appears to be down')
    } else if (error.code === 'ETIMEDOUT') {
      errorResponse += "I'm taking longer than usual to think. Can you try again?"
      console.log('ERROR: FastAPI request timed out')
    } else if (error.response) {
      errorResponse += `There was an issue with my AI processing (${error.response.status}).`
      console.log('ERROR: FastAPI returned error:', error.response.status, error.response.data)
    }
    
    res.json({ 
      response: errorResponse,
      error: true,
      debug_info: {
        error_type: error.code || 'unknown',
        fastapi_url: process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
      }
    })
  }
})

module.exports = r