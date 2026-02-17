# Frontend Integration Guide - Conversation Memory

## Critical Requirement

For conversation memory to work, **the frontend MUST send `conversation_history` with every API request** (except the first question).

## The Problem

If your conversation memory isn't working, it's because:
1. ❌ Frontend sends each question as a new isolated request
2. ❌ Backend receives no conversation history
3. ❌ Backend treats each question as brand new
4. ❌ No park context detected, no memory maintained

## The Solution

**Send conversation history with each request:**

```javascript
// Maintain conversation history in state
const [conversationHistory, setConversationHistory] = useState([]);

async function sendMessage(question) {
  const response = await fetch('http://your-backend/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: question,
      conversation_history: conversationHistory  // ← CRITICAL!
    })
  });

  const data = await response.json();

  // Update history with new exchange
  setConversationHistory([
    ...conversationHistory,
    { role: 'user', content: question },
    { role: 'assistant', content: data.answer }
  ]);

  return data;
}
```

## Complete React Example

```javascript
import React, { useState } from 'react';

function NationalParksChatbot() {
  const [messages, setMessages] = useState([]);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const API_URL = 'http://localhost:8000/api/chat';

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    // Add user message to display
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage,
          top_k: 5,
          conversation_history: conversationHistory  // ← Send history
        })
      });

      const data = await response.json();

      // Add assistant response to display
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      }]);

      // Update conversation history (keep last 20 messages)
      const newHistory = [
        ...conversationHistory,
        { role: 'user', content: userMessage },
        { role: 'assistant', content: data.answer }
      ].slice(-20);  // Keep only last 20 messages

      setConversationHistory(newHistory);

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'error',
        content: 'Failed to get response'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const clearConversation = () => {
    setMessages([]);
    setConversationHistory([]);
  };

  return (
    <div className="chatbot">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <p>{msg.content}</p>
            {msg.sources && (
              <div className="sources">
                Sources: {msg.sources.map(s => s.park_name).join(', ')}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about national parks..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? 'Sending...' : 'Send'}
        </button>
        <button onClick={clearConversation}>
          New Conversation
        </button>
      </div>

      <div className="debug-info">
        History: {conversationHistory.length} messages
      </div>
    </div>
  );
}

export default NationalParksChatbot;
```

## Python Example (for testing)

```python
import requests

API_URL = "http://localhost:8000/api/chat"
conversation_history = []

def ask_question(question):
    global conversation_history

    response = requests.post(API_URL, json={
        "question": question,
        "conversation_history": conversation_history
    })

    result = response.json()
    answer = result['answer']

    # Update history
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    # Keep only last 20 messages
    conversation_history = conversation_history[-20:]

    return answer

# Usage
answer1 = ask_question("Tell me about Glacier National Park")
print(answer1)

answer2 = ask_question("What wildlife can I see there?")
print(answer2)  # Should understand "there" = Glacier

answer3 = ask_question("When should I camp there?")
print(answer3)  # Should maintain Glacier context
```

## API Request Format

### First Question (no history)
```json
{
  "question": "Tell me about Glacier National Park",
  "top_k": 5
}
```

### Follow-up Questions (WITH history)
```json
{
  "question": "What wildlife can I see there?",
  "top_k": 5,
  "conversation_history": [
    {
      "role": "user",
      "content": "Tell me about Glacier National Park"
    },
    {
      "role": "assistant",
      "content": "Glacier National Park is located in Montana..."
    }
  ]
}
```

## Important Notes

1. **Always include previous exchanges**
   - Each request should include ALL previous user/assistant messages
   - Max 20 messages (10 exchanges) to stay within token limits

2. **Message format**
   - Each message MUST have `role` ("user" or "assistant")
   - Each message MUST have `content` (the text)

3. **Managing history**
   - Keep history in frontend state (useState, Redux, etc.)
   - Limit to 20 messages (slice oldest if exceeding)
   - Clear history when user starts "New Conversation"

4. **Don't send system messages**
   - Only send "user" and "assistant" roles
   - Backend adds system prompts automatically

## Testing Your Frontend

Run the test script to verify backend is working:
```bash
python test_conversation_backend.py
```

If backend tests pass but your app doesn't work, the issue is in your frontend.

Check:
- ✅ Are you storing conversation_history in state?
- ✅ Are you sending it with EVERY request?
- ✅ Are you updating it after each response?
- ✅ Are messages in correct format (role + content)?

## Debugging

Add console logging to verify:
```javascript
console.log('Sending conversation history:', conversationHistory);
```

Check backend logs for:
```
INFO: Conversation history provided: True
INFO: History length: 4 messages
INFO: ✓ Park detected from conversation: glac
INFO: Active park code for search: glac
```

If you see "Conversation history provided: False", your frontend isn't sending history.

## Common Mistakes

❌ **Sending empty history**
```javascript
conversation_history: []  // This won't help!
```

❌ **Only sending last message**
```javascript
conversation_history: [lastMessage]  // Need ALL messages!
```

❌ **Wrong message format**
```javascript
{ user: "question", bot: "answer" }  // Use "role" and "content"!
```

✅ **Correct format**
```javascript
[
  { role: "user", content: "Tell me about Glacier" },
  { role: "assistant", content: "Glacier is in Montana..." },
  { role: "user", content: "What wildlife is there?" },
  { role: "assistant", content: "Glacier has grizzly bears..." }
]
```

## Need Help?

If conversation memory still doesn't work after implementing this:
1. Run `python test_conversation_backend.py` to verify backend
2. Check browser console for conversation_history being sent
3. Check backend logs for park detection messages
4. Verify API requests include conversation_history in the payload
