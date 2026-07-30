#!/bin/bash

# ==========================================================================
# BiliRAG Quickstart Orchestrator Script (macOS / Linux)
# Helps spawn the system in either full-featured Docker containerized mode
# or a portable, zero-config local Python environment.
# ==========================================================================

# Text styling
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

clear
echo -e "${CYAN}${BOLD}"
echo "======================================================================"
echo "    ____  _ ___  ___   ______   ____  ____ _  _______  ____  ______   "
echo "   / __ )(_) (_) __ \ / ____/  / __ )/ __ \ |/ / ___/ / __ \/ ____/   "
echo "  / __  / / / / /_/ // / __   / __  / /_/ /   /\__ \ / /_/ / /        "
echo " / /_/ / / / / _, _// /_/ /  / /_/ / _, _/   /___/ // _, _/ /___      "
echo "/_____/_/_/_/_/ |_| \____/  /_____/_/ |_/_/|__/____//_/ |_|\____/      "
echo "                                                                      "
echo "        Bilingual (Urdu/English) RAG Support & MLOps System           "
echo "======================================================================"
echo -e "${RESET}"

echo -e "This script assists in launching the RAG Chatbot & Telemetry Dashboard."
echo -e "Choose your preferred deployment strategy below:"
echo ""
echo -e "  ${BOLD}[1] Docker Compose Mode (Recommended - Full Production Stack)${RESET}"
echo -e "      Spawns FastAPI, Celery Workers, Redis broker, and SQLite logs"
echo -e "      with full background task offloading and concurrency."
echo ""
echo -e "  ${BOLD}[2] Local Native Mode (Single Process - Zero Docker Required)${RESET}"
echo -e "      Sets up a Python virtual environment, installs packages, and"
echo -e "      runs the API + Frontend on a thread-safe synchronous fallback."
echo ""
echo -ne "${BOLD}Select execution mode [1 or 2]: ${RESET}"
read -r MODE

if [ "$MODE" = "1" ]; then
    echo -e "\n${BLUE}--> Validating Docker and Docker-Compose dependencies...${RESET}"
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[Error] Docker is not installed or not running. Please launch Docker Desktop or select Local Native Mode.${RESET}"
        exit 1
    fi
    
    echo -e "${GREEN}[Success] Docker daemon verified.${RESET}"
    echo -e "${BLUE}--> Launching full containerized production stack via docker-compose...${RESET}"
    echo -e "${YELLOW}Note: Initial download of models and PyTorch may take a few minutes on first build.${RESET}\n"
    
    docker-compose up --build
    
elif [ "$MODE" = "2" ]; then
    echo -e "\n${BLUE}--> Spawning Native Python environment...${RESET}"
    
    # 1. Navigate to backend directory
    cd backend || exit 1
    
    # 2. Setup python virtual environment
    if [ ! -d "venv" ]; then
        echo -e "${BLUE}--> Creating python virtual environment 'venv'...${RESET}"
        python3 -m venv venv
    fi
    
    # 3. Activate virtual environment
    echo -e "${BLUE}--> Activating virtual environment...${RESET}"
    source venv/bin/activate
    
    # 4. Install requirements
    echo -e "${BLUE}--> Installing dependencies from requirements.txt...${RESET}"
    echo -e "${YELLOW}Note: Standard PyTorch and transformers are being installed. This may take a moment.${RESET}"
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # 5. Launch FastAPI application
    echo -e "\n${GREEN}${BOLD}======================================================================"
    echo "    BiliRAG Local API is running at: http://localhost:8000/"
    echo "    Opening your web browser now... Press Ctrl+C to terminate."
    echo -e "======================================================================${RESET}\n"
    
    # Open browser automatically on mac
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sleep 2 && open "http://localhost:8000/" &
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sleep 2 && xdg-open "http://localhost:8000/" &
    fi
    
    # Run uvicorn
    uvicorn app.main:app --host 127.0.0.1 --port 8000
    
else
    echo -e "${RED}Invalid option selected. Exiting.${RESET}"
    exit 1
fi
