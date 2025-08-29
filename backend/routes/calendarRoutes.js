// Google calendar integration for MAI - enhanced for recommendations
const express = require('express')
const { google } = require('googleapis')
const r = express.Router()

// Connect to Google APIs
const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  process.env.GOOGLE_REDIRECT_URI
)

// Send user to Google login page
r.get('/auth', (req, res) => {
  const username = req.query.username || 'guest'
  const returnType = req.query.return || 'basic'
  
  const authUrl = oauth2Client.generateAuthUrl({
    access_type: 'offline',
    scope: ['https://www.googleapis.com/auth/calendar'],
    state: JSON.stringify({ username, returnType }) // encode extra info for the callback
  })
  
  console.log(`Redirecting ${username} to Google auth for ${returnType}`)
  res.redirect(authUrl)
})

// Google sends user back here after login
r.get('/callback', async (req, res) => {
  const { code, state } = req.query
  
  try {
    // Parse state information
    let stateData = { username: 'guest', returnType: 'basic' }
    try {
      stateData = JSON.parse(state)
    } catch (e) {
      console.log('Could not parse state, using defaults')
    }
    
    console.log('Processing calendar callback for:', stateData)
    
    // Exchange code for access token
    const { tokens } = await oauth2Client.getToken(code)
    oauth2Client.setCredentials(tokens)
    
    // Save user's tokens (in real app you'd save to database)
    req.session = req.session || {}
    req.session.googleTokens = tokens
    req.session.username = stateData.username
    
    console.log('Calendar tokens saved for user:', stateData.username)
    
    // Redirect based on return type
    if (stateData.returnType === 'calendar_add') {
      // Redirect back to frontend with special flag for adding pending items
      res.redirect('http://localhost:5500?calendar=connected&action=add_pending')
    } else {
      // Standard calendar connection
      res.redirect('http://localhost:5500?calendar=connected')
    }
    
  } catch (e) {
    console.log('Calendar auth failed:', e.message)
    res.redirect('http://localhost:5500?calendar=error')
  }
})

// Add single or multiple reminders to calendar
r.post('/add-reminder', async (req, res) => {
  const { title, description, datetime, username, items } = req.body
  
  console.log('Add reminder request:', { title, items: items?.length || 'single', username })
  
  // Make sure they're logged in with Google first
  if (!req.session?.googleTokens) {
    console.log('No Google tokens found, redirecting to auth')
    return res.json({ 
      success: false, 
      authUrl: `/api/calendar/auth?username=${username}&return=calendar_add`
    })
  }
  
  try {
    // Set up calendar connection
    oauth2Client.setCredentials(req.session.googleTokens)
    const calendar = google.calendar({ version: 'v3', auth: oauth2Client })
    
    const results = []
    
    // Handle multiple items (from AI recommendations)
    if (items && Array.isArray(items)) {
      console.log(`Adding ${items.length} recommendation items to calendar`)
      
      for (let i = 0; i < items.length; i++) {
        const item = items[i]
        
        // Create event for each recommendation - spread over consecutive days
        const eventTime = new Date()
        eventTime.setDate(eventTime.getDate() + i + 1) // spread over consecutive days
        eventTime.setHours(10, 0, 0, 0) // 10 AM each day
        
        const endTime = new Date(eventTime)
        endTime.setMinutes(endTime.getMinutes() + 30) // 30 minute duration
        
        const event = {
          summary: `MAI Wellness: ${item}`,
          description: `Wellness reminder from MAI: ${item}\n\nThis reminder was generated based on your conversation with MAI to support your mental health and wellbeing.`,
          start: {
            dateTime: eventTime.toISOString(),
            timeZone: 'America/New_York'
          },
          end: {
            dateTime: endTime.toISOString(),
            timeZone: 'America/New_York'
          },
          reminders: {
            useDefault: false,
            overrides: [
              { method: 'popup', minutes: 15 },
              { method: 'email', minutes: 60 }
            ]
          },
          colorId: '2' // sage green color for wellness
        }
        
        try {
          const result = await calendar.events.insert({
            calendarId: 'primary',
            resource: event
          })
          
          results.push({
            success: true,
            item: item,
            eventId: result.data.id,
            eventLink: result.data.htmlLink
          })
          
          console.log(`Successfully added: ${item}`)
          
        } catch (itemError) {
          console.error(`Failed to add item "${item}":`, itemError.message)
          results.push({
            success: false,
            item: item,
            error: itemError.message
          })
        }
      }
      
      const successCount = results.filter(r => r.success).length
      const totalCount = results.length
      
      res.json({ 
        success: successCount > 0,
        results: results,
        summary: `Added ${successCount} out of ${totalCount} reminders to your calendar`,
        successCount: successCount,
        totalCount: totalCount
      })
      
    } else {
      // Handle single reminder (original functionality)
      console.log('Adding single reminder to calendar')
      
      const event = {
        summary: title || 'MAI wellness reminder',
        description: description || 'Time for a quick mental health check-in',
        start: {
          dateTime: datetime || new Date(Date.now() + 60*60*1000).toISOString(),
          timeZone: 'America/New_York'
        },
        end: {
          dateTime: datetime || new Date(Date.now() + 60*60*1000 + 30*60*1000).toISOString(),
          timeZone: 'America/New_York'
        },
        reminders: {
          useDefault: false,
          overrides: [
            { method: 'popup', minutes: 15 },
            { method: 'email', minutes: 30 }
          ]
        },
        colorId: '2'
      }
      
      const result = await calendar.events.insert({
        calendarId: 'primary',
        resource: event
      })
      
      res.json({ 
        success: true, 
        eventId: result.data.id,
        eventLink: result.data.htmlLink
      })
    }
    
  } catch (e) {
    console.error('Calendar error:', e.message)
    res.json({ 
      success: false, 
      error: 'Could not add to calendar. Please try reconnecting your Google Calendar.',
      details: e.message
    })
  }
})

// Add multiple recommendations endpoint
r.post('/add-recommendations', async (req, res) => {
  const { recommendations, username } = req.body
  
  if (!recommendations || !Array.isArray(recommendations) || recommendations.length === 0) {
    return res.json({ success: false, error: 'No recommendations provided' })
  }
  
  console.log(`Adding ${recommendations.length} recommendations for ${username}`)
  
  // Make sure they're logged in first
  if (!req.session?.googleTokens) {
    return res.json({ 
      success: false, 
      authUrl: `/api/calendar/auth?username=${username}&return=calendar_add`
    })
  }
  
  try {
    oauth2Client.setCredentials(req.session.googleTokens)
    const calendar = google.calendar({ version: 'v3', auth: oauth2Client })
    
    const results = []
    const baseDate = new Date()
    baseDate.setDate(baseDate.getDate() + 1) // start tomorrow
    
    for (let i = 0; i < recommendations.length; i++) {
      const recommendation = recommendations[i]
      
      // Schedule each recommendation on consecutive days
      const eventDate = new Date(baseDate)
      eventDate.setDate(eventDate.getDate() + i)
      eventDate.setHours(10, 0, 0, 0) // 10 AM
      
      const endDate = new Date(eventDate)
      endDate.setMinutes(endDate.getMinutes() + 30)
      
      const event = {
        summary: `MAI Wellness: ${recommendation}`,
        description: `Wellness recommendation from MAI: ${recommendation}\n\nThis suggestion was generated based on your conversation with MAI to support your mental health and wellbeing.`,
        start: {
          dateTime: eventDate.toISOString(),
          timeZone: 'America/New_York'
        },
        end: {
          dateTime: endDate.toISOString(),
          timeZone: 'America/New_York'
        },
        reminders: {
          useDefault: false,
          overrides: [
            { method: 'popup', minutes: 15 },
            { method: 'email', minutes: 60 }
          ]
        },
        colorId: '10' // basil green for wellness
      }
      
      try {
        const result = await calendar.events.insert({
          calendarId: 'primary',
          resource: event
        })
        
        results.push({
          success: true,
          recommendation: recommendation,
          date: eventDate.toLocaleDateString(),
          eventId: result.data.id
        })
        
      } catch (error) {
        results.push({
          success: false,
          recommendation: recommendation,
          error: error.message
        })
      }
    }
    
    const successCount = results.filter(r => r.success).length
    
    res.json({
      success: successCount > 0,
      results: results,
      successCount: successCount,
      totalCount: recommendations.length,
      message: `Successfully added ${successCount} out of ${recommendations.length} wellness reminders to your calendar.`
    })
    
  } catch (error) {
    console.error('Error adding recommendations:', error)
    res.json({
      success: false,
      error: 'Failed to add recommendations to calendar',
      details: error.message
    })
  }
})

// Check what reminders they have coming up
r.get('/reminders/:username', async (req, res) => {
  if (!req.session?.googleTokens) {
    return res.json({ reminders: [], needsAuth: true })
  }
  
  try {
    oauth2Client.setCredentials(req.session.googleTokens)
    const calendar = google.calendar({ version: 'v3', auth: oauth2Client })
    
    // Look for events in next 2 weeks
    const now = new Date()
    const twoWeeksFromNow = new Date(now.getTime() + 14*24*60*60*1000)
    
    const response = await calendar.events.list({
      calendarId: 'primary',
      timeMin: now.toISOString(),
      timeMax: twoWeeksFromNow.toISOString(),
      q: 'MAI', // only MAI events
      singleEvents: true,
      orderBy: 'startTime'
    })
    
    const reminders = response.data.items.map(event => ({
      id: event.id,
      title: event.summary,
      description: event.description,
      datetime: event.start.dateTime || event.start.date,
      link: event.htmlLink,
      isWellnessReminder: event.summary?.includes('MAI Wellness') || false
    }))
    
    console.log(`Found ${reminders.length} MAI reminders for ${req.params.username}`)
    
    res.json({ reminders, needsAuth: false })
    
  } catch (e) {
    console.log('Could not get reminders:', e.message)
    res.json({ reminders: [], error: 'Could not get reminders' })
  }
})

// Check calendar connection status
r.get('/status/:username', async (req, res) => {
  const hasTokens = !!req.session?.googleTokens
  
  res.json({
    connected: hasTokens,
    username: req.session?.username || null,
    needsAuth: !hasTokens
  })
})

module.exports = r