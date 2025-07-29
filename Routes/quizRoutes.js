const express=require('express')
const Quiz=require('../models/Quiz')
const r=express.Router()

// save quiz
r.post('/save-inline',async(req,res)=>{
  const { userId, quizzes } = req.body
  try{
    const results=[]
    for (let q of quizzes){
      const doc=new Quiz({
        userId,type:q.type,answers:q.answers,summary:q.summary,createdAt:new Date()
      })
      await doc.save()
      results.push(doc)
    }
    res.json({status:'saved',results})
  }catch(e){
    res.status(500).json({error:'fail'})
  }
})

module.exports=r
