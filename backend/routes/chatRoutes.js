const express=require('express')
const axios=require('axios')
const User=require('../models/User')
const r=express.Router()

//handle the inputs and the stuff for chats 
r.post('/',async(req,res)=>{
 const {userId,user_input}=req.body
 const user=await User.findOne({username:userId})
 const quiz_summary=user && user.quizResults.length
  ? user.quizResults.map(q=>q.summary).join(', ')
  : "No quiz data yet"

 try{
   const ai=await axios.post('http://127.0.0.1:8000/generate',{user_input,quiz_summary})
   res.json(ai.data)
 }catch(e){
   console.error('fast api fail',e)
   res.json({response:"sorry having trouble"})
 }
})

module.exports=r
