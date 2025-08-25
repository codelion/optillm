"""
Test utilities for OptILLM tests
Provides common functions and constants for consistent testing
"""

import os
import sys
import time
import subprocess
import platform
from typing import Optional
from openai import OpenAI

# Standard test model for all tests - small and fast
TEST_MODEL = "google/gemma-3-270m-it"
TEST_MODEL_MLX = "mlx-community/gemma-3-270m-it-bf16"

def setup_test_env():
    """Set up test environment with local inference"""
    os.environ["OPTILLM_API_KEY"] = "optillm"
    return TEST_MODEL

def get_test_client(base_url: str = "http://localhost:8000/v1") -> OpenAI:
    """Get OpenAI client configured for local optillm"""
    return OpenAI(api_key="optillm", base_url=base_url)

def is_mlx_available():
    """Check if MLX is available (macOS only)"""
    if platform.system() != "Darwin":
        return False
    try:
        from optillm.inference import MLX_AVAILABLE
        return MLX_AVAILABLE
    except ImportError:
        return False

def start_test_server(model: str = TEST_MODEL, port: int = 8000) -> subprocess.Popen:
    """
    Start optillm server for testing
    Returns the process handle
    """
    # Set environment for local inference
    env = os.environ.copy()
    env["OPTILLM_API_KEY"] = "optillm"
    
    # Start server
    proc = subprocess.Popen([
        sys.executable, "optillm.py",
        "--model", model,
        "--port", str(port)
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    
    return proc

def stop_test_server(proc: subprocess.Popen):
    """Stop the test server"""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

def get_simple_test_messages():
    """Get simple test messages for basic validation"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one word."}
    ]

def get_math_test_messages():
    """Get math test messages for reasoning validation"""
    return [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
    ]

def get_thinking_test_messages():
    """Get test messages that should generate thinking tokens"""
    return [
        {"role": "system", "content": "Think step by step and use <think></think> tags."},
        {"role": "user", "content": "What is 3 * 4? Show your thinking."}
    ]