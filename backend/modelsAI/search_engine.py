import re
from .model_loader import KNOWLEDGE_CORPUS

def improved_search(query, top_k=3):
    if not KNOWLEDGE_CORPUS:
        return []

    query_lower = query.lower()
    search_terms = set(query_lower.split())

    if any(w in query_lower for w in ["nervous", "anxious", "worried", "stress"]):
        search_terms.update(["anxiety", "calm", "breathing", "relax"])
    if any(w in query_lower for w in ["conference", "presentation", "work", "meeting"]):
        search_terms.update(["professional", "confidence", "performance"])
    if any(w in query_lower for w in ["doctor", "medical", "health", "appointment"]):
        search_terms.update(["care", "treatment"])
    if any(w in query_lower for w in ["week", "busy", "schedule", "time"]):
        search_terms.update(["overwhelmed", "planning", "organization"])

    results = []
    for i, entry in enumerate(KNOWLEDGE_CORPUS):
        text_content = ""
        if isinstance(entry, dict):
            for key in ['text', 'content', 'response', 'answer', 'advice', 'data', 'message']:
                if key in entry and entry[key]:
                    text_content = str(entry[key]).lower()
                    break
        elif isinstance(entry, str):
            text_content = entry.lower()
        elif isinstance(entry, list) and entry:
            text_content = str(entry[0]).lower()

        if text_content:
            text_words = set(re.findall(r'\b\w+\b', text_content))
            overlap = len(search_terms & text_words)
            semantic_score = sum(1 for t in search_terms if t in text_content)
            total_score = overlap + semantic_score
            if total_score > 0:
                results.append({
                    'content': text_content,
                    'original_entry': entry,
                    'score': total_score / len(search_terms),
                    'index': i
                })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def use_best_training_content(user_input, relevant_docs):
    if not relevant_docs:
        return "I want to help you with that. Can you tell me more about what's specifically concerning you?"

    best_match = relevant_docs[0]['content']
    if len(best_match) < 50 and len(relevant_docs) > 1:
        best_match += " " + relevant_docs[1]['content']

    if len(best_match) > 400:
        sentences = best_match.split('.')
        best_match = '. '.join(sentences[:2]) + '.'

    intro = "I hear what you're going through. "
    if any(word in user_input.lower() for word in ["nervous", "anxious", "conference"]):
        intro = "I understand that nervousness before important events. "
    elif any(word in user_input.lower() for word in ["doctor", "medical", "waiting"]):
        intro = "Waiting for medical news can be really stressful. "
    elif any(word in user_input.lower() for word in ["week", "busy", "ahead"]):
        intro = "Big weeks can feel overwhelming. "

    return intro + best_match
