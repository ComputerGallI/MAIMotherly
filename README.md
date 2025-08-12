# MAI – Motherly AI 

**A warm, emotionally intelligent AI that checks in with you, listens, and gently nudges you toward wellness to give you a better outlook on your future.**

---

## What is MAI?
MAI stands for **Motherly AI** – an app that acts like a mom, without the generational trauma!   
You can:
- **Chat** with MAI about your day, finances, and anything in your life that has meaning. MAI can respond with your mental health in mind.  
- **Take short quizzes** to help MAI understand your personality, stress level, love language, and what makes you "tick" 
- **Get gentle reminders** like “get 30 mins of sunlight” or “schedule a massage” — and add them to your Google Calendar so you can actually implement them into your life. 

The goal is to  make you feel heard, understood, and supported.

---

## Target Audience

- A kind daily check-in
- Easy going emotional wellness tracking
- Easy, actionable self-care reminders that make your life better. 

---

##  How MAI Grabs info
We use **RAG (Retrieval-Augmented Generation)** with **affect computing** to give responses that feel right for you.

**Tech Stack:**
- **Frontend:** React, HTML, CSS, JavaScript
- **Backend:** Node.js + Express
- **AI Service:** FastAPI running BERT (retrieval) + BART (generation)
- **Database:** MongoDB Atlas
- **Model Training:** Google Colab with datasets from Kaggle & GoEmotions, Mental Health counseling conversations from Huggingface

**Main Flow:**
1. **User Login/Register** Info stored securely in MongoDB  
2. **Quiz or Chat**  Quizzes saved in MongoDB or localStorage  
3. **AI Processing** RAG pipeline retrieves and generates responses  
4. **Reminders & Visualization**  Results shown as charts; reminders can be added to Google Calendar  

---
