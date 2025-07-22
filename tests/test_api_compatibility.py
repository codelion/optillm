#!/usr/bin/env python3
"""
Test API compatibility with OpenAI format
"""

import pytest
import os
from openai import OpenAI
import json


@pytest.fixture
def client():
    """Create OpenAI client for optillm proxy"""
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "test-key"),
        base_url="http://localhost:8000/v1"
    )


def test_basic_completion(client):
    """Test basic chat completion"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=10
    )
    
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], 'message')
    assert hasattr(response.choices[0].message, 'content')


def test_n_parameter(client):
    """Test n parameter for multiple completions"""
    n = 3
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a one-line joke"}
        ],
        n=n,
        temperature=0.8,
        max_tokens=50
    )
    
    assert len(response.choices) == n
    # Check all responses are different (with high temperature)
    contents = [choice.message.content for choice in response.choices]
    assert len(set(contents)) > 1  # At least some different responses


def test_approach_prefix(client):
    """Test approach prefix in model name"""
    response = client.chat.completions.create(
        model="moa-gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        max_tokens=10
    )
    
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0


def test_extra_body_approach(client):
    """Test approach specification via extra_body"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        extra_body={"optillm_approach": "bon"},
        max_tokens=10
    )
    
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0


def test_streaming(client):
    """Test streaming response"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Count from 1 to 5"}
        ],
        stream=True,
        max_tokens=50
    )
    
    chunks = list(stream)
    assert len(chunks) > 0
    # First chunk should have role
    assert chunks[0].choices[0].delta.role == "assistant"
    # Later chunks should have content
    content_chunks = [chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content]
    assert len(content_chunks) > 0


if __name__ == "__main__":
    # Run basic tests if pytest not available
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "test-key"),
        base_url="http://localhost:8000/v1"
    )
    
    print("Running basic API compatibility tests...")
    
    try:
        test_basic_completion(client)
        print("✅ Basic completion test passed")
    except Exception as e:
        print(f"❌ Basic completion test failed: {e}")
    
    try:
        test_n_parameter(client)
        print("✅ N parameter test passed")
    except Exception as e:
        print(f"❌ N parameter test failed: {e}")
    
    try:
        test_approach_prefix(client)
        print("✅ Approach prefix test passed")
    except Exception as e:
        print(f"❌ Approach prefix test failed: {e}")
    
    print("\nDone!")