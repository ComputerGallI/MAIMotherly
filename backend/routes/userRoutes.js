const express=require('express')
const User=require('../models/User')
const r=express.Router()

// seed users
r.get('/seed',async(req,res)=>{
  await User.insertMany([
    {username:'IndieMovies',password:'12345!',quizResults:[],suggestions:[]},
    {username:'Ryanator',password:'12345!',quizResults:[],suggestions:[]}
  ])
  res.send('seeded users')
})

// login
r.post('/login',async(req,res)=>{
  const{username,password}=req.body
  const u=await User.findOne({username})
  if(!u||u.password!==password)return res.status(401).json({error:'invalid'})
  res.json(u)
})

module.exports=r
