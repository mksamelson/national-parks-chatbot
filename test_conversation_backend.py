"""
Test script to verify conversation memory is working in the backend

This script tests the backend API directly with conversation history
to verify that park context detection and memory are working correctly.

Run this script while the backend server is running:
  cd backend
  uvicorn main:app --reload

Then in another terminal:
  python test_conversation_backend.py
"""
import requests
import json

API_URL = "http://localhost:8000/api/chat"

print("="*70)
print("TESTING CONVERSATION MEMORY - Backend API")
print("="*70)

# Test 1: First question about Glacier
print("\n" + "="*70)
print("TEST 1: First question about Glacier")
print("="*70)

request1 = {
    "question": "Tell me about Glacier National Park"
}

print(f"\nRequest:\n{json.dumps(request1, indent=2)}")

try:
    response1 = requests.post(API_URL, json=request1, timeout=30)
    response1.raise_for_status()
    result1 = response1.json()

    print(f"\n✓ Response received")
    print(f"Answer preview: {result1['answer'][:200]}...")
    print(f"Sources: {len(result1['sources'])} from parks: {set(s['park_name'] for s in result1['sources'])}")

except Exception as e:
    print(f"\n✗ Request failed: {e}")
    print("\nMake sure the backend server is running:")
    print("  cd backend")
    print("  uvicorn main:app --reload")
    exit(1)

# Test 2: Follow-up with conversation history
print("\n" + "="*70)
print("TEST 2: Follow-up question WITH conversation history")
print("="*70)

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

print(f"\nRequest:")
print(f"  Question: '{request2['question']}'")
print(f"  Conversation history: {len(request2['conversation_history'])} messages")

try:
    response2 = requests.post(API_URL, json=request2, timeout=30)
    response2.raise_for_status()
    result2 = response2.json()

    print(f"\n✓ Response received")
    print(f"Answer preview: {result2['answer'][:300]}...")

    # Check if answer mentions Glacier
    answer_lower = result2['answer'].lower()
    mentions_glacier = 'glacier' in answer_lower

    print(f"\nPark-specific sources: {set(s['park_name'] for s in result2['sources'])}")
    print(f"Answer mentions Glacier: {mentions_glacier}")

    if mentions_glacier and any('glacier' in s['park_name'].lower() for s in result2['sources']):
        print("\n✅ TEST PASSED - Backend correctly understood 'there' = Glacier")
    else:
        print("\n⚠️  TEST WARNING - Backend may not have detected Glacier context")
        print("Check the backend logs for park detection messages")

except Exception as e:
    print(f"\n✗ Request failed: {e}")
    exit(1)

# Test 3: Another follow-up
print("\n" + "="*70)
print("TEST 3: Another follow-up question")
print("="*70)

request3 = {
    "question": "When is the best time to camp there?",
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

print(f"\nRequest:")
print(f"  Question: '{request3['question']}'")
print(f"  Conversation history: {len(request3['conversation_history'])} messages")

try:
    response3 = requests.post(API_URL, json=request3, timeout=30)
    response3.raise_for_status()
    result3 = response3.json()

    print(f"\n✓ Response received")
    print(f"Answer preview: {result3['answer'][:300]}...")

    # Check if answer is about camping at Glacier
    answer_lower = result3['answer'].lower()
    has_glacier = 'glacier' in answer_lower
    has_camping_info = any(word in answer_lower for word in ['camp', 'summer', 'season', 'june', 'july'])

    print(f"\nAnswer mentions Glacier: {has_glacier}")
    print(f"Answer has camping info: {has_camping_info}")

    # Check if LLM said it doesn't know which park
    confused = any(phrase in answer_lower for phrase in [
        "which park",
        "doesn't specify",
        "not sure which park",
        "don't know which park"
    ])

    if confused:
        print("\n❌ TEST FAILED - LLM is confused about which park")
        print("The LLM said it doesn't know which park you're referring to")
        print("\nThis means the park context isn't being passed to the LLM correctly.")
        print("Check backend logs for park detection messages.")
    elif has_glacier and has_camping_info:
        print("\n✅ TEST PASSED - Backend maintains Glacier context correctly")
    else:
        print("\n⚠️  TEST INCONCLUSIVE - Check the full answer")

except Exception as e:
    print(f"\n✗ Request failed: {e}")
    exit(1)

print("\n" + "="*70)
print("BACKEND TESTING COMPLETE")
print("="*70)
print("\nIf tests passed, the backend is working correctly.")
print("If you're still having issues in your app, check that your frontend")
print("is sending conversation_history with each API request.")
print("\nTo check backend logs, look for these messages:")
print("  - 'Conversation history provided: True'")
print("  - '✓ Park detected from conversation: glac'")
print("  - 'Active park code for search: glac'")
