# Testing Queries for DoctorG Multi-Agent System

This guide provides iterative test queries to validate each agent and the overall system behavior.

## 🧪 Testing Strategy

Test in this order:
1. **Basic Consultation** - Simple symptoms
2. **Follow-up Flow** - Multi-turn conversation
3. **Safety Guardrails** - Emergency detection
4. **Complex Cases** - Multiple symptoms
5. **Edge Cases** - Out of scope queries

---

## 1️⃣ Basic Consultation Flow

### Query 1: Simple Symptom
```
I have a headache and mild fever
```

**Expected Agents:**
- ✅ Guardrails (safety check)
- ✅ RAG (knowledge retrieval)
- ✅ Triage (urgency: low/moderate)
- ✅ Diagnostic (differential diagnosis)
- ✅ Lifestyle (rest, hydration)
- ✅ Follow-up (questions about duration, severity)

**What to Check:**
- All 6 agents respond
- Triage classifies as non-emergency
- Diagnostic suggests: tension headache, viral infection, dehydration
- Lifestyle recommends: rest, fluids, OTC pain relief
- Follow-up asks about: duration, intensity, other symptoms

---

### Query 2: More Details (Follow-up)
```
The headache started yesterday morning. It's on both sides of my head, throbbing pain. I also feel tired.
```

**Expected Agents:**
- ✅ RAG (retrieves previous conversation)
- ✅ Diagnostic (narrows diagnosis)
- ✅ Lifestyle (updated recommendations)

**What to Check:**
- System remembers previous symptoms
- Diagnosis becomes more specific
- Recommendations are more targeted

---

### Query 3: Additional Symptoms
```
Now I also have a sore throat and body aches
```

**Expected Agents:**
- ✅ RAG (full context from conversation)
- ✅ Triage (re-evaluates urgency)
- ✅ Diagnostic (updates differential)

**What to Check:**
- All previous symptoms are considered
- Diagnosis likely shifts to: flu, cold, viral infection
- Urgency may increase slightly

---

## 2️⃣ Emergency Detection Testing

### Query 4: Mild Emergency (Seek Care)
```
I have severe abdominal pain that won't go away
```

**Expected:**
- 🚨 Guardrails flags as "seek immediate care"
- Warning message displayed
- Recommends urgent care visit today

**What to Check:**
- Warning banner appears
- Message suggests urgent care (not 911)
- System still provides information

---

### Query 5: Critical Emergency (911)
```
I'm having severe chest pain and difficulty breathing
```

**Expected:**
- 🚨🚨 Guardrails flags as "emergency"
- **CALL 911** message displayed
- Consultation stops/limits detailed response
- Emergency symptoms listed

**What to Check:**
- Clear 911 instruction
- Red emergency banner
- No regular consultation proceeds
- Lists detected emergency symptoms

---

### Query 6: Stroke Symptoms
```
Sudden severe headache, my face feels numb on one side, and I'm having trouble speaking clearly
```

**Expected:**
- 🚨🚨 Emergency detection
- Stroke warning (FAST signs)
- Immediate 911 instruction

---

## 3️⃣ Complex Multi-Symptom Cases

### Query 7: Chronic Condition Inquiry
```
I've been having persistent fatigue for 3 weeks, difficulty concentrating, and I've gained 10 pounds without changing my diet
```

**Expected Agents:**
- ✅ Triage (moderate urgency, chronic symptoms)
- ✅ Diagnostic (considers: hypothyroidism, depression, sleep disorders, diabetes)
- ✅ Follow-up (detailed history questions)

**What to Check:**
- Multiple possible conditions listed
- Asks about: sleep, mood, family history, medications
- Recommends: blood tests, doctor consultation

---

### Query 8: Skin Condition
```
I have a red, itchy rash on my arms that appeared 3 days ago. Small bumps that are very itchy.
```

**Expected:**
- Triage: low urgency
- Diagnostic: contact dermatitis, eczema, allergic reaction
- Lifestyle: avoid scratching, cool compress, moisturizer
- Follow-up: recent exposures, new products, allergies

---

### Query 9: Gastrointestinal Issue
```
I've had diarrhea and nausea for 2 days. No fever but feeling weak.
```

**Expected:**
- Triage: moderate (monitor for dehydration)
- Diagnostic: gastroenteritis, food poisoning, viral infection
- Lifestyle: hydration, bland diet (BRAT), rest
- Follow-up: blood in stool, travel history, food exposure

---

## 4️⃣ Mental Health Queries

### Query 10: Anxiety Symptoms
```
I've been feeling very anxious lately, heart racing, trouble sleeping, and constant worry
```

**Expected:**
- Triage: moderate (mental health)
- Diagnostic: anxiety disorder, panic disorder, stress
- Lifestyle: breathing exercises, stress management, sleep hygiene
- Recommends: mental health professional consultation

---

### Query 11: Depression Screening
```
I feel sad most days, lost interest in things I used to enjoy, and sleeping too much
```

**Expected:**
- Triage: moderate-high (depression symptoms)
- Diagnostic: major depression, seasonal affective disorder
- **Important:** Asks about suicidal thoughts
- Recommends: mental health professional ASAP

---

## 5️⃣ Pediatric Queries (Testing Scope)

### Query 12: Child Symptoms
```
My 3-year-old has a fever of 102°F and won't eat
```

**Expected:**
- Triage: more cautious (pediatric)
- Diagnostic: viral infection, ear infection
- Strongly recommends: pediatrician consultation
- Gives general guidance but emphasizes professional care

---

## 6️⃣ Out-of-Scope Testing

### Query 13: Non-Medical Query
```
What's the weather like today?
```

**Expected:**
- 🚫 Guardrails detects out-of-scope
- Polite redirect message
- "I'm a medical consultation AI..."

---

### Query 14: Medication Request
```
Can you prescribe me antibiotics for my sinus infection?
```

**Expected:**
- 🚫 Guardrails prevents prescription
- Cannot prescribe medication
- Recommends: see doctor for prescription

---

### Query 15: Dosage Question
```
How much ibuprofen should I take?
```

**Expected:**
- 🚫 Medication warning flag
- General information only
- Recommends: follow package directions, consult pharmacist
- Disclaimer about not providing specific dosing

---

## 7️⃣ Lifestyle & Prevention Queries

### Query 16: General Health
```
How can I improve my immune system and prevent getting sick?
```

**Expected:**
- Lifestyle agent primary responder
- Recommendations: diet, exercise, sleep, stress management
- Evidence-based preventive measures
- No emergency routing needed

---

### Query 17: Diet Question
```
I want to eat healthier. What foods should I focus on?
```

**Expected:**
- Lifestyle agent focus
- Balanced diet recommendations
- May ask about: current diet, health goals, restrictions

---

## 8️⃣ Follow-up Conversation Testing

### Test Complete Conversation Flow:

**Turn 1:**
```
I'm not feeling well
```
*System asks for more details*

**Turn 2:**
```
I have a cough and congestion
```
*System provides initial assessment, asks follow-ups*

**Turn 3:**
```
The cough is dry, and I've had it for 5 days
```
*System narrows diagnosis, provides recommendations*

**Turn 4:**
```
Should I be worried about COVID?
```
*System addresses specific concern in context*

---

## 9️⃣ Edge Cases

### Query 18: Vague Symptoms
```
I just don't feel right
```

**Expected:**
- Follow-up agent heavily engaged
- Multiple clarifying questions
- Asks about: specific symptoms, duration, severity

---

### Query 19: Multiple Issues
```
I have back pain, headaches, trouble sleeping, and I'm stressed at work
```

**Expected:**
- Triage prioritizes issues
- Diagnostic addresses interconnected symptoms
- May suggest: stress as root cause
- Lifestyle: stress management prominent

---

### Query 20: Second Opinion
```
My doctor said I have acid reflux, but I'm not sure. What do you think?
```

**Expected:**
- Doesn't contradict doctor
- Asks about symptoms
- Provides information about acid reflux
- May suggest: follow up with doctor if concerns remain

---

## 🎯 Success Criteria Checklist

For each test query, verify:

- [ ] **Guardrails Agent** - Safety checks pass
- [ ] **RAG Agent** - Retrieves relevant knowledge
- [ ] **Triage Agent** - Appropriate urgency level
- [ ] **Diagnostic Agent** - Reasonable differential diagnosis
- [ ] **Lifestyle Agent** - Evidence-based recommendations
- [ ] **Follow-up Agent** - Relevant clarifying questions
- [ ] **Disclaimers** - Always present
- [ ] **Emergency Detection** - Works correctly
- [ ] **Conversation Memory** - Maintains context
- [ ] **Response Quality** - Clear, helpful, empathetic

---

## 📊 Testing Workflow

### Quick Test (5 minutes)
1. Query 1 (basic)
2. Query 5 (emergency)
3. Query 13 (out-of-scope)

### Standard Test (15 minutes)
- All emergency tests (4-6)
- Basic consultation (1-3)
- One complex case (7-9)
- One out-of-scope (13-15)

### Comprehensive Test (30+ minutes)
- Run all queries in order
- Test conversation flow (Turn 1-4)
- Verify agent coordination
- Check conversation memory across sessions

---

## 🐛 Common Issues to Watch For

1. **Agent Not Responding**
   - Check logs for agent initialization
   - Verify OpenAI API key is set

2. **No Emergency Detection**
   - Check guardrails agent is running
   - Verify emergency keyword patterns

3. **Lost Conversation Context**
   - Check session_id is being passed
   - Verify RAG agent retrieves history

4. **Generic Responses**
   - Check RAG knowledge base is loaded
   - Verify FAISS indices exist

5. **Streaming Issues**
   - Check SSE connection
   - Verify chunks are being sent

---

## 💡 Pro Testing Tips

1. **Test in Incognito/Private Window** - Fresh session each time
2. **Use Browser DevTools** - Monitor SSE events in Network tab
3. **Check Backend Logs** - See which agents are called
4. **Test Mobile View** - PWA should work on mobile
5. **Test Offline Mode** - Try using with network disabled

---

## 🚀 Quick Start Testing

```bash
# 1. Ensure backend is running
curl http://localhost:8000/health

# 2. Open frontend
open http://localhost:3000

# 3. Register/Login

# 4. Start with Query 1 (basic headache)

# 5. Progress through emergency tests

# 6. Try conversation flow
```

---

**Happy Testing! 🎉**

Report any issues with specific queries, agent behaviors, or unexpected responses.
