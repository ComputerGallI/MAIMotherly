// import stuff we need
const request = require('supertest')   // to fake api calls
const express = require('express')     // to make a mini app for test
const mongoose = require('mongoose')   // db connect
const Quiz = require('../models/Quiz') // quiz model
const quizRoutes = require('../routes/quizRoutes') // our routes

// make small app just for testing
const app = express()
app.use(express.json())
app.use('/api/quiz', quizRoutes)

// connect to a test db before tests start
beforeAll(async () => {
  // using local test db so we dont mess main db
  await mongoose.connect('mongodb://127.0.0.1:27017/mai_test', {
    useNewUrlParser: true,
    useUnifiedTopology: true
  })
})

// after tests are done clean db and close
afterAll(async () => {
  await mongoose.connection.db.dropDatabase()  // delete test db
  await mongoose.connection.close()            // close connection
})

describe('Quiz API', () => {

  // happy path test
  test('save quiz works', async () => {
    // fake sending quiz to server
    const res = await request(app)
      .post('/api/quiz/save-inline')
      .send({
        userId: 'TestUser',
        quizzes: [
          { type: 'Personality', answers: { q1: 'Yes' }, summary: 'Friendly' }
        ]
      })

    // expect to see 200 ok
    expect(res.statusCode).toBe(200)
    // expect server say saved
    expect(res.body.status).toBe('saved')
  })

  // edge case test
  test('send empty data gives error', async () => {
    // fake to send empty
    const res = await request(app).post('/api/quiz/save-inline').send({})

    // expect fail code
    expect(res.statusCode).toBe(500)
  })
})
