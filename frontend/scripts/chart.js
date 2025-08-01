fetch('/api/quiz/latest-summary?userId=IndieMovies')
.then(r=>r.json())
.then(d=>{
  let ctx=document.getElementById('quizChart')
  new Chart(ctx,{
    type:'bar',
    data:{
      labels:d.types||["Personality","Stress","Love","21Q"],
      datasets:[{
        label:"Scores",
        data:d.scores||[1,2,3,4],
        backgroundColor:'rgba(153,102,255,0.6)'
      }]
    },
    options:{responsive:true,scales:{y:{beginAtZero:true}}}
  })
})
