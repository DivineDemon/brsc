#!/bin/bash
# ==========================================================================
# React TS Frontend Build & Deployment Script
# Installs dependencies, compiles TypeScript/React code, and copies static
# assets to the root directory for Hugging Face Static Space deployment.
# ==========================================================================

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
RESET="\033[0m"

echo -e "${BLUE}${BOLD}--> Navigating to frontend directory...${RESET}"
cd frontend

echo -e "${BLUE}${BOLD}--> Installing npm dependencies...${RESET}"
npm install

echo -e "${BLUE}${BOLD}--> Building production React + TypeScript + Tailwind bundle...${RESET}"
npm run build

echo -e "${BLUE}${BOLD}--> Copying static assets from frontend/dist/ to root directory...${RESET}"
cd ..
cp -r frontend/dist/* .

echo -e "${GREEN}${BOLD}✅ Build complete! Production React TypeScript app is ready at root.${RESET}"
