const express = require('express')
const User = require('../models/User')
const r = express.Router()

// seed some demo users
r.get('/seed', async (req,res)=>{
  await User.deleteMany({})
  await User.insertMany([
    {username:'IndieMovies',email:'indie@demo.com',password:'12345!'},
    {username:'Ryanator',email:'ryan@demo.com',password:'12345!'}
  ])
  res.json({ok:true})
})

// register new user
r.post('/register', async (req,res)=>{
  const {username,password,email} = req.body
  if(!username || !password) return res.status(400).json({error:'need username and password'})
  
  const exists = await User.findOne({username})
  if(exists) return res.json({ok:false,msg:'username taken'})
  
  await User.create({username,password,email})
  res.json({ok:true})
})

// login existing user
r.post('/login', async (req,res)=>{
  const {username,password} = req.body
  const user = await User.findOne({username})
  
  if(!user || user.password!==password) {
    return res.json({error:'invalid credentials'})
  }
  
  res.json({username:user.username,email:user.email})
})

module.exports = r