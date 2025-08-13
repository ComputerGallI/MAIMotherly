const express = require('express')
const r = express.Router()

// get reminders (demo for now)
r.get('/', async (req,res)=>{
  res.json([])
})

// add reminder (demo for now)
r.post('/', async (req,res)=>{
  res.json({ok:true})
})

module.exports = r