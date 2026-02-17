"""
Test script to verify the PRODUCTION API has conversation memory features

This tests the live Render deployment to check if the backend has been
updated with conversation memory, park detection, and query rewriting.
"""
import requests
import json

PROD_API_URL = "https://national-parks-chatbot.onrender.com/api/chat"

print("="*70)
print("TESTING PRODUCTION API - Conversation Memory")
print("="*70)

# Test 1: Check if API is responding
print("\n" + "="*70)
print("TEST 1: API Health Check")
print("="*70)

try:
    health_response = requests.get("https://national-parks-chatbot.onrender.com/health", timeout=10)
    print(f"✓ API is responding: {health_response.status_code}")
    print(f"Response: {health_response.json()}")
except Exception as e:
    print(f"✗ API health check failed: {e}")
    print("\nThe backend might be down or starting up.")
    print("Render free tier apps sleep after inactivity - first request wakes them up.")
    exit(1)

# Test 2: First question about Glacier
print("\n" + "="*70)
print("TEST 2: First question about Glacier")
print("="*70)

request1 = {
    "question": "Tell me about Glacier National Park"
}

print(f"Sending: {json.dumps(request1, indent=2)}")

try:
    response1 = requests.post(PROD_API_URL, json=request1, timeout=120)
    response1.raise_for_status()
    result1 = response1.json()

    print(f"\n✓ Response received")
    print(f"Answer preview: {result1['answer'][:200]}...")
    print(f"Sources: {set(s['park_name'] for s in result1['sources'])}")

except Exception as e:
    print(f"\n✗ Request failed: {e}")
    exit(1)

# Test 3: Follow-up WITH conversation history
print("\n" + "="*70)
print("TEST 3: Follow-up question WITH conversation history")
print("="*70)
print("This tests if the backend has conversation memory features")

request2 = {
    "question": "What wildlife can I see there?",
    "conversation_history": [
        {
            "role": "user",
            "content": "Tell me about Glacier National Park"
        },
        {
            "role": "assistant",
            "content": result1['answer']
        }
    ]
}

print(f"\nSending:")
print(f"  Question: '{request2['question']}'")
print(f"  History: {len(request2['conversation_history'])} messages")

try:
    response2 = requests.post(PROD_API_URL, json=request2, timeout=120)
    response2.raise_for_status()
    result2 = response2.json()

    print(f"\n✓ Response received")
    print(f"Answer preview: {result2['answer'][:300]}...")

    # Analyze the response
    sources_parks = set(s['park_name'] for s in result2['sources'])
    answer_lower = result2['answer'].lower()

    print(f"\nSource parks: {sources_parks}")
    print(f"Answer mentions Glacier: {'glacier' in answer_lower}")

    # Check if sources are all from Glacier
    glacier_only = all('glacier' in p.lower() for p in sources_parks)

    if glacier_only and 'glacier' in answer_lower:
        print("\n✅ PASS - Backend correctly focused on Glacier!")
        print("Conversation memory IS working in production.")
    else:
        print("\n❌ FAIL - Backend returned info from other parks")
        print("Conversation memory NOT working in production.")
        print("\nThis means either:")
        print("1. Render hasn't deployed the latest backend code yet")
        print("2. There's an issue with the deployment")
        print("\nCheck Render dashboard: https://dashboard.render.com")
        print("Look for the 'national-parks-chatbot-api' service")
        print("Check if there's a deployment in progress or failed")

except Exception as e:
    print(f"\n✗ Request failed: {e}")
    exit(1)

# Test 4: Another vague follow-up
print("\n" + "="*70)
print("TEST 4: Very vague follow-up")
print("="*70)

request3 = {
    "question": "When should I camp there?",
    "conversation_history": [
        {
            "role": "user",
            "content": "Tell me about Glacier National Park"
        },
        {
            "role": "assistant",
            "content": result1['answer']
        },
        {
            "role": "user",
            "content": "What wildlife can I see there?"
        },
        {
            "role": "assistant",
            "content": result2['answer']
        }
    ]
}

print(f"Question: '{request3['question']}'")
print(f"History: {len(request3['conversation_history'])} messages")

try:
    response3 = requests.post(PROD_API_URL, json=request3, timeout=120)
    response3.raise_for_status()
    result3 = response3.json()

    print(f"\n✓ Response received")
    print(f"Answer preview: {result3['answer'][:300]}...")

    answer_lower = result3['answer'].lower()
    sources_parks = set(s['park_name'] for s in result3['sources'])

    print(f"\nSource parks: {sources_parks}")

    # Check for confusion
    confused = any(phrase in answer_lower for phrase in [
        "which park",
        "doesn't specify",
        "not sure which",
        "don't know which"
    ])

    glacier_focused = all('glacier' in p.lower() for p in sources_parks)

    if confused:
        print("\n❌ FAIL - Backend is confused about which park")
        print("Says it doesn't know which park you're referring to")
    elif glacier_focused:
        print("\n✅ PASS - Backend maintains Glacier context!")
    else:
        print("\n⚠️  PARTIAL - Answer includes other parks")

except Exception as e:
    print(f"\n✗ Request failed: {e}")

print("\n" + "="*70)
print("PRODUCTION API TEST COMPLETE")
print("="*70)
print("\nIf tests failed, the production backend needs to be redeployed.")
print("\nTo trigger redeployment on Render:")
print("1. Go to: https://dashboard.render.com")
print("2. Find 'national-parks-chatbot-api' service")
print("3. Click 'Manual Deploy' -> 'Deploy latest commit'")
print("4. Wait for deployment to complete (~2-5 minutes)")
print("5. Run this test again")
