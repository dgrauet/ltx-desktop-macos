#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   LTX Desktop macOS — Environment Setup  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# 1. Check macOS version >= 14.0 (Sonoma)
echo -e "${BLUE}Checking macOS version...${NC}"
MACOS_VERSION=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
if [ "$MACOS_MAJOR" -lt 14 ]; then
    echo -e "${RED}✗ macOS $MACOS_VERSION detected. Minimum required: macOS 14.0 (Sonoma).${NC}"
    echo -e "${RED}  MLX requires macOS 14.0+. Please update your system.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ macOS $MACOS_VERSION${NC}"

# 2. Check Apple Silicon (arm64)
echo -e "${BLUE}Checking architecture...${NC}"
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo -e "${RED}✗ Architecture: $ARCH. MLX requires Apple Silicon (arm64).${NC}"
    echo -e "${RED}  This app cannot run on Intel Macs.${NC}"
    exit 1
fi
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
echo -e "${GREEN}✓ $CHIP ($ARCH)${NC}"

# 3. Check/install Homebrew
echo -e "${BLUE}Checking Homebrew...${NC}"
if command -v brew &>/dev/null; then
    echo -e "${GREEN}✓ Homebrew installed$(brew --version | head -1 | sed 's/Homebrew //' | sed 's/^/ (v/' | sed 's/$/)/')${NC}"
else
    echo -e "${YELLOW}⚠ Homebrew not found. Install it with:${NC}"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo -e "${YELLOW}  Then re-run this script.${NC}"
    exit 1
fi

# 4. Check/install Python 3.12+
echo -e "${BLUE}Checking Python 3.12+...${NC}"
PYTHON_CMD=""
for cmd in python3.12 python3.13 python3; do
    if command -v "$cmd" &>/dev/null; then
        PY_VERSION=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 12 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${YELLOW}⚠ Python 3.12+ not found. Installing via brew...${NC}"
    brew install python@3.12
    PYTHON_CMD="python3.12"
fi
echo -e "${GREEN}✓ $($PYTHON_CMD --version)${NC}"

# 5. Check/install uv
echo -e "${BLUE}Checking uv (Python package manager)...${NC}"
if command -v uv &>/dev/null; then
    echo -e "${GREEN}✓ uv $(uv --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)${NC}"
else
    echo -e "${YELLOW}⚠ uv not found. Installing...${NC}"
    if command -v brew &>/dev/null; then
        brew install uv
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    echo -e "${GREEN}✓ uv installed${NC}"
fi

# 6. Check/install ffmpeg
echo -e "${BLUE}Checking ffmpeg...${NC}"
if command -v ffmpeg &>/dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    echo -e "${GREEN}✓ ffmpeg $FFMPEG_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ ffmpeg not found. Installing via brew...${NC}"
    brew install ffmpeg
    echo -e "${GREEN}✓ ffmpeg installed${NC}"
fi

# 7. Install Python dependencies
echo ""
echo -e "${BLUE}Installing Python dependencies...${NC}"
cd "$PROJECT_ROOT/backend"
uv sync
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# 8. System info summary
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           System Information              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
RAM_GB=$((RAM_BYTES / 1073741824))
echo -e "  Chip:       ${GREEN}$CHIP${NC}"
echo -e "  RAM:        ${GREEN}${RAM_GB}GB${NC}"
echo -e "  macOS:      ${GREEN}$MACOS_VERSION${NC}"
echo -e "  Python:     ${GREEN}$($PYTHON_CMD --version)${NC}"
echo -e "  uv:         ${GREEN}$(uv --version 2>&1)${NC}"
echo -e "  ffmpeg:     ${GREEN}$(ffmpeg -version 2>&1 | head -1)${NC}"

# RAM recommendation
if [ "$RAM_GB" -lt 32 ]; then
    echo ""
    echo -e "${YELLOW}⚠ ${RAM_GB}GB RAM detected. 32GB+ recommended for comfortable generation.${NC}"
    echo -e "${YELLOW}  Consider using cloud text encoding for lower memory usage.${NC}"
fi

# 9. Next steps
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Download models:  ./scripts/download_models.sh"
echo "  2. Start dev:        ./scripts/dev.sh"
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
