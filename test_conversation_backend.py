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
import sys
import io
import requests
import json

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

API_URL = "http://localhost:8002/api/chat"

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

# Test 4: Assistant messages don't change park context (CRITICAL)
print("\n" + "="*70)
print("TEST 4: Assistant mentions of other parks don't change context")
print("="*70)

request4 = {
    "question": "What are the hiking trails?",
    "conversation_history": [
        {
            "role": "user",
            "content": "Tell me about Glacier National Park"
        },
        {
            "role": "assistant",
            "content": "Glacier National Park is similar to Yellowstone and Yosemite in many ways. Like Yellowstone, it has diverse wildlife including bears and elk."
        }
    ]
}

print(f"\nRequest:")
print(f"  Question: '{request4['question']}'")
print(f"  User mentioned: Glacier")
print(f"  Assistant mentioned: Yellowstone, Yosemite")
print(f"  Expected park context: Glacier (from user, not assistant)")

try:
    response4 = requests.post(API_URL, json=request4, timeout=30)
    response4.raise_for_status()
    result4 = response4.json()

    print(f"\n✓ Response received")
    park_sources = set(s['park_name'] for s in result4['sources'])
    print(f"Sources from parks: {park_sources}")

    # Should return results from Glacier, NOT Yellowstone/Yosemite
    has_glacier = any('glacier' in s['park_name'].lower() for s in result4['sources'])
    has_yellowstone = any('yellowstone' in s['park_name'].lower() for s in result4['sources'])
    has_yosemite = any('yosemite' in s['park_name'].lower() for s in result4['sources'])

    if has_glacier and not has_yellowstone and not has_yosemite:
        print("\n✅ TEST PASSED - Context stayed with Glacier (user's mention)")
    else:
        print("\n❌ TEST FAILED - Context changed to wrong park")
        print("Assistant messages should NOT change park context!")

except Exception as e:
    print(f"\n✗ Request failed: {e}")

# Test 5: Park in current question overrides history
print("\n" + "="*70)
print("TEST 5: Park in current question overrides history")
print("="*70)

request5 = {
    "question": "What about Zion National Park?",
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
print(f"  Question: '{request5['question']}'")
print(f"  Previous context: Glacier")
print(f"  Current question mentions: Zion")
print(f"  Expected park context: Zion (current question overrides)")

try:
    response5 = requests.post(API_URL, json=request5, timeout=30)
    response5.raise_for_status()
    result5 = response5.json()

    print(f"\n✓ Response received")
    park_sources = set(s['park_name'] for s in result5['sources'])
    print(f"Sources from parks: {park_sources}")

    has_zion = any('zion' in s['park_name'].lower() for s in result5['sources'])
    has_glacier = any('glacier' in s['park_name'].lower() for s in result5['sources'])

    if has_zion and not has_glacier:
        print("\n✅ TEST PASSED - Switched to Zion as requested")
    else:
        print("\n❌ TEST FAILED - Should have switched to Zion")

except Exception as e:
    print(f"\n✗ Request failed: {e}")

# Test 6: Most recent user park mention takes precedence
print("\n" + "="*70)
print("TEST 6: Most recent user park mention takes precedence")
print("="*70)

request6 = {
    "question": "Tell me more about hiking trails",
    "conversation_history": [
        {
            "role": "user",
            "content": "What's the weather like at Yellowstone?"
        },
        {
            "role": "assistant",
            "content": "Yellowstone has varied weather..."
        },
        {
            "role": "user",
            "content": "How about Yosemite National Park?"
        },
        {
            "role": "assistant",
            "content": "Yosemite has a Mediterranean climate..."
        }
    ]
}

print(f"\nRequest:")
print(f"  Question: '{request6['question']}'")
print(f"  User mentioned: Yellowstone (older), then Yosemite (recent)")
print(f"  Expected park context: Yosemite (most recent user mention)")

try:
    response6 = requests.post(API_URL, json=request6, timeout=30)
    response6.raise_for_status()
    result6 = response6.json()

    print(f"\n✓ Response received")
    park_sources = set(s['park_name'] for s in result6['sources'])
    print(f"Sources from parks: {park_sources}")

    has_yosemite = any('yosemite' in s['park_name'].lower() for s in result6['sources'])
    has_yellowstone = any('yellowstone' in s['park_name'].lower() for s in result6['sources'])

    if has_yosemite and not has_yellowstone:
        print("\n✅ TEST PASSED - Used most recent park (Yosemite)")
    else:
        print("\n❌ TEST FAILED - Should use most recent park mention")

except Exception as e:
    print(f"\n✗ Request failed: {e}")

# Test 7: Extended conversation (5+ turns) maintains context
print("\n" + "="*70)
print("TEST 7: Extended conversation maintains park context")
print("="*70)

request7 = {
    "question": "Are there visitor centers?",
    "conversation_history": [
        {
            "role": "user",
            "content": "Tell me about Crater Lake"
        },
        {
            "role": "assistant",
            "content": "Crater Lake is a beautiful volcanic lake..."
        },
        {
            "role": "user",
            "content": "What wildlife is there?"
        },
        {
            "role": "assistant",
            "content": "You can see deer, eagles, and other wildlife..."
        },
        {
            "role": "user",
            "content": "What about hiking?"
        },
        {
            "role": "assistant",
            "content": "There are many trails around the rim..."
        }
    ]
}

print(f"\nRequest:")
print(f"  Question: '{request7['question']}'")
print(f"  Initial park mentioned: Crater Lake (6 messages ago)")
print(f"  Follow-up turns: 3 (with pronouns only)")
print(f"  Expected park context: Crater Lake")

try:
    response7 = requests.post(API_URL, json=request7, timeout=30)
    response7.raise_for_status()
    result7 = response7.json()

    print(f"\n✓ Response received")
    park_sources = set(s['park_name'] for s in result7['sources'])
    print(f"Sources from parks: {park_sources}")

    has_crater = any('crater' in s['park_name'].lower() for s in result7['sources'])

    if has_crater:
        print("\n✅ TEST PASSED - Maintained context through extended conversation")
    else:
        print("\n❌ TEST FAILED - Lost park context after multiple turns")

except Exception as e:
    print(f"\n✗ Request failed: {e}")

print("\n" + "="*70)
print("COMPREHENSIVE TESTING COMPLETE")
print("="*70)
print("\nAll tests completed. Review results above.")
print("\nKey improvements tested:")
print("  ✓ Assistant messages don't change park context")
print("  ✓ Current question overrides conversation history")
print("  ✓ Most recent USER park mention takes precedence")
print("  ✓ Extended conversations maintain context")
print("\nTo check backend logs, look for these messages:")
print("  - '✓ Park in current question: <park> (<code>)'")
print("  - '✓ Park from user message: <park> (<code>)'")
print("  - '✗ No park detected in question or recent user messages'")
