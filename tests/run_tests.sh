#!/bin/bash
# Run all tests for OptILLM

set -e  # Exit on error

echo "Running OptILLM Tests"
echo "===================="

# Check if optillm server is running
check_server() {
    curl -s http://localhost:8000/v1/health > /dev/null 2>&1
}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Python version:"
python --version

# Install test dependencies if needed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -r tests/requirements.txt
fi

# Check if optillm server is running
if ! check_server; then
    echo -e "${YELLOW}Warning: OptILLM server not detected at localhost:8000${NC}"
    echo "Some integration tests may fail. Start the server with: python optillm.py"
    echo ""
fi

# Run unit tests
echo -e "\n${GREEN}Running unit tests...${NC}"
python -m pytest tests/test_plugins.py -v

# Run API tests if server is available
if check_server; then
    echo -e "\n${GREEN}Running API compatibility tests...${NC}"
    python -m pytest tests/test_api_compatibility.py -v
else
    echo -e "\n${YELLOW}Skipping API tests (server not running)${NC}"
fi

# Run n parameter test
if check_server; then
    echo -e "\n${GREEN}Running n parameter test...${NC}"
    python tests/test_n_parameter.py
else
    echo -e "\n${YELLOW}Skipping n parameter test (server not running)${NC}"
fi

# Run main test suite with a simple test
echo -e "\n${GREEN}Running main test suite (simple test only)...${NC}"
cd "$(dirname "$0")/.."  # Go to project root
python tests/test.py --approaches none bon --single-test "Simple Math Problem"

echo -e "\n${GREEN}All tests completed!${NC}"