// handle quiz answers (localStorage for demo)
function saveQuizResult(type,score){
  let results=JSON.parse(localStorage.getItem('quizResults'))||{}
  results[type]=score
  localStorage.setItem('quizResults',JSON.stringify(results))
}
