const chatWindow=document.getElementById('chatWindow')
const chatInput=document.getElementById('chatInput')
const sendBtn=document.getElementById('sendChat')

function appendChat(sender,text){
  let div=document.createElement('div')
  div.className=sender
  div.textContent=text
  chatWindow.appendChild(div)
  chatWindow.scrollTop=chatWindow.scrollHeight
}

appendChat('bot',"Hello, I can't wait to get to know you. Take a quiz or talk to me?")

sendBtn.onclick=()=>{
  const msg=chatInput.value
  if(!msg.trim())return
  appendChat('user',msg)
  chatInput.value=''

  fetch('/api/chat',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({userId:'IndieMovies',user_input:msg})
  })
  .then(r=>r.json())
  .then(d=>appendChat('bot',d.response||'...'))
}

function startQuiz(type){
  appendChat('bot',`Starting the ${type} quiz now!`)
}
