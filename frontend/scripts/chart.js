// chart for quiz results
document.addEventListener('DOMContentLoaded',()=>{
  let ctx=document.getElementById('resultsChart')
  if(!ctx)return

  new Chart(ctx,{
    type:'radar',
    data:{
      labels:['Personality','Stress','Love','21Q'],
      datasets:[{
        label:'Your Results',
        data:[0,0,0,0],
        backgroundColor:'rgba(174,120,255,0.3)',
        borderColor:'#ae78ff'
      }]
    },
    options:{
      scales:{r:{beginAtZero:true,max:10}}
    }
  })
})
