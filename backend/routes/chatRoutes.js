const express = require('express')
const axios = require('axios')
const ChatLog = require('../models/ChatLog')
const r = express.Router()

// chat with mai
r.post('/', async (req,res)=>{
  const { userId, user_input } = req.body
  
  try{
    // send to fastapi for ai response
    const ai = await axios.post(`${process.env.FASTAPI_URL}/generate`, {
      user_input,
      quiz_summary: buildQuizSummary(userId) // placeholder for now
    })
    
    const data = ai.data || {}
    
    // save chat to database
    await ChatLog.create({
      username: userId || 'guest',
      input: user_input,
      response: data.response || '',
      retrieved: data.retrieved_docs || []
    })
    
    res.json(data)
  }catch(e){
    console.error('chat error', e.message)
    res.json({response:"sorry, having trouble right now"})
  }
})

function buildQuizSummary(userId){
  // simple stub - in real app you'd get this from user profile
  return 'personality and quiz data will go here'
}

module.exports = r