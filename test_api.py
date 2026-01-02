"""
Example test script for the AI Trip Planner API
Run this after starting the server to test the endpoint
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_generate_itinerary():
    """Test the generate-itinerary endpoint"""
    
    # Test data
    payload = {
        "video_url": "https://www.instagram.com/reel/DNPyyw4I_8B/?igsh=MTI3MDZkMjM2NWZ4MQ==",
        "caption_text": ""
    }
    
    print("Testing POST /generate-itinerary...")
    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        # Make request
        response = requests.post(
            f"{BASE_URL}/generate-itinerary",
            json=payload,
            timeout=30
        )
        
        # Check response
        if response.status_code == 200:
            print(f"\n✅ Success! Status code: {response.status_code}")
            print(f"\nResponse:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"\n❌ Error! Status code: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection error! Make sure the server is running:")
        print("   uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

def test_health_check():
    """Test the root health check endpoint"""
    
    print("Testing GET /...")
    
    try:
        response = requests.get(BASE_URL)
        
        if response.status_code == 200:
            print(f"✅ Success! Status code: {response.status_code}")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Error! Status code: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection error! Make sure the server is running.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("AI Trip Planner - API Test Script")
    print("=" * 60)
    
    # Test health check
    test_health_check()
    
    print("\n" + "=" * 60 + "\n")
    
    # Test main endpoint
    test_generate_itinerary()
    
    print("\n" + "=" * 60)
