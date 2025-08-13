// google calendar stuff for mai
const express = require('express')
const { google } = require('googleapis')
const r = express.Router()

// connect to google apis
const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  process.env.GOOGLE_REDIRECT_URI
)

// send user to google login page
r.get('/auth', (req, res) => {
  const authUrl = oauth2Client.generateAuthUrl({
    access_type: 'offline',
    scope: ['https://www.googleapis.com/auth/calendar'],
    state: req.query.username || 'guest' // remember who this is
  })
  res.redirect(authUrl)
})

// google sends user back here after login
r.get('/callback', async (req, res) => {
  const { code, state } = req.query
  
  try {
    // swap code for actual token
    const { tokens } = await oauth2Client.getToken(code)
    oauth2Client.setCredentials(tokens)
    
    // save user's tokens (in real app you'd save to database)
    req.session = req.session || {}
    req.session.googleTokens = tokens
    req.session.username = state
    
    // send them back to the main page
    res.redirect('http://localhost:3000?calendar=connected')
  } catch (e) {
    console.log('login failed:', e.message)
    res.redirect('http://localhost:3000?calendar=error')
  }
})

// actually add something to their calendar
r.post('/add-reminder', async (req, res) => {
  const { title, description, datetime, username } = req.body
  
  // make sure they're logged in first
  if (!req.session?.googleTokens) {
    return res.json({ 
      success: false, 
      authUrl: `/api/calendar/auth?username=${username}`
    })
  }
  
  try {
    // set up calendar connection
    oauth2Client.setCredentials(req.session.googleTokens)
    const calendar = google.calendar({ version: 'v3', auth: oauth2Client })
    
    // make the calendar event
    const event = {
      summary: title || 'MAI wellness reminder',
      description: description || 'time for a quick mental health check-in',
      start: {
        dateTime: datetime || new Date(Date.now() + 60*60*1000).toISOString(), // 1 hour from now
        timeZone: 'America/New_York'
      },
      end: {
        dateTime: datetime || new Date(Date.now() + 60*60*1000 + 30*60*1000).toISOString(), // 30 min long
        timeZone: 'America/New_York'
      },
      reminders: {
        useDefault: false,
        overrides: [
          { method: 'popup', minutes: 15 },
          { method: 'email', minutes: 30 }
        ]
      }
    }
    
    // actually add it
    const result = await calendar.events.insert({
      calendarId: 'primary',
      resource: event
    })
    
    res.json({ 
      success: true, 
      eventId: result.data.id,
      eventLink: result.data.htmlLink
    })
    
  } catch (e) {
    console.log('calendar broke:', e.message)
    res.json({ success: false, error: 'couldn\'t add to calendar' })
  }
})

// check what reminders they have coming up
r.get('/reminders/:username', async (req, res) => {
  if (!req.session?.googleTokens) {
    return res.json({ reminders: [], needsAuth: true })
  }
  
  try {
    oauth2Client.setCredentials(req.session.googleTokens)
    const calendar = google.calendar({ version: 'v3', auth: oauth2Client })
    
    // look for events in next week
    const now = new Date()
    const weekFromNow = new Date(now.getTime() + 7*24*60*60*1000)
    
    const response = await calendar.events.list({
      calendarId: 'primary',
      timeMin: now.toISOString(),
      timeMax: weekFromNow.toISOString(),
      q: 'MAI', // only mai events
      singleEvents: true,
      orderBy: 'startTime'
    })
    
    const reminders = response.data.items.map(event => ({
      id: event.id,
      title: event.summary,
      description: event.description,
      datetime: event.start.dateTime || event.start.date,
      link: event.htmlLink
    }))
    
    res.json({ reminders, needsAuth: false })
    
  } catch (e) {
    console.log('couldn\'t get reminders:', e.message)
    res.json({ reminders: [], error: 'couldn\'t get reminders' })
  }
})

module.exports = r