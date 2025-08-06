// simple chat
async function sendMessage(){
  let input=document.getElementById('userInput')
  let msg=input.value.trim()
  if(!msg)return
  addMessage('You: '+msg)
  input.value=''

  let res=await fetch('http://localhost:5000/api/chat',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({userId:'IndieMovies',user_input:msg})
  })
  let data=await res.json()
  addMessage('MAI: '+data.response)
}

// helper to show msg
function addMessage(text){
  let win=document.getElementById('chat-window')
  let div=document.createElement('div')
  div.textContent=text
  win.appendChild(div)
  win.scrollTop=win.scrollHeight
}
