#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_PID=""

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    LTX Desktop macOS — Dev Environment    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${YELLOW}Stopping backend (PID $BACKEND_PID)...${NC}"
        kill "$BACKEND_PID" 2>/dev/null
        # Wait up to 5 seconds for graceful shutdown
        for i in $(seq 1 10); do
            if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        # Force kill if still running
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo -e "${YELLOW}Force-killing backend...${NC}"
            kill -9 "$BACKEND_PID" 2>/dev/null || true
        fi
        echo -e "${GREEN}✓ Backend stopped${NC}"
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

for cmd in python3 uv ffmpeg; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}✗ $cmd not found. Run ./scripts/setup.sh first.${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ All prerequisites found${NC}"

# 2. Check if backend already running on port 8000
echo -e "${BLUE}Checking port 8000...${NC}"
if lsof -ti:8000 &>/dev/null; then
    echo -e "${YELLOW}⚠ Port 8000 already in use. Checking if it's our backend...${NC}"
    if curl -sf http://127.0.0.1:8000/api/v1/system/health &>/dev/null; then
        echo -e "${GREEN}✓ Backend already running${NC}"
        BACKEND_PID=$(lsof -ti:8000 | head -1)
    else
        echo -e "${RED}✗ Port 8000 in use by another process. Free it and retry.${NC}"
        exit 1
    fi
else
    # 3. Start backend
    echo -e "${BLUE}Starting backend...${NC}"
    cd "$PROJECT_ROOT/backend"
    uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload &
    BACKEND_PID=$!
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✓ Backend started (PID $BACKEND_PID)${NC}"

    # 4. Wait for health check
    echo -e "${BLUE}Waiting for backend to be ready...${NC}"
    for i in $(seq 1 60); do
        if curl -sf http://127.0.0.1:8000/api/v1/system/health &>/dev/null; then
            echo -e "${GREEN}✓ Backend is ready${NC}"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo -e "${RED}✗ Backend failed to start within 30s${NC}"
            cleanup
            exit 1
        fi
        sleep 0.5
    done
fi

# 5. Open Xcode project if it exists
if [ -d "$PROJECT_ROOT/app/LTXDesktop.xcodeproj" ]; then
    echo -e "${BLUE}Opening Xcode project...${NC}"
    open "$PROJECT_ROOT/app/LTXDesktop.xcodeproj"
    echo -e "${GREEN}✓ Xcode opened${NC}"
else
    echo -e "${YELLOW}⚠ Xcode project not found at app/LTXDesktop.xcodeproj${NC}"
fi

# 6. Status
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Dev Environment Ready           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "  Backend:  ${GREEN}http://127.0.0.1:8000${NC}"
echo -e "  Health:   ${GREEN}http://127.0.0.1:8000/api/v1/system/health${NC}"
echo -e "  Docs:     ${GREEN}http://127.0.0.1:8000/docs${NC}"
echo -e "  PID:      ${GREEN}$BACKEND_PID${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the backend${NC}"

# 7. Wait for backend process
wait "$BACKEND_PID" 2>/dev/null || true
