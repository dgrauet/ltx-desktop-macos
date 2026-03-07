#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    LTX Desktop macOS — Model Download     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# 1. Check disk space
echo -e "${BLUE}Checking disk space...${NC}"
AVAILABLE_GB=$(df -g "$HOME" | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_GB" -lt 50 ]; then
    echo -e "${YELLOW}⚠ Only ${AVAILABLE_GB}GB free. Models require ~50GB total.${NC}"
    echo -e "${YELLOW}  Proceeding anyway — downloads can be resumed if interrupted.${NC}"
else
    echo -e "${GREEN}✓ ${AVAILABLE_GB}GB available${NC}"
fi

# 2. Check huggingface-cli
echo -e "${BLUE}Checking huggingface-cli...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_HF="$SCRIPT_DIR/../backend/.venv/bin/huggingface-cli"

if command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
    echo -e "${GREEN}✓ huggingface-cli available${NC}"
elif [ -f "$VENV_HF" ]; then
    HF_CLI="$VENV_HF"
    echo -e "${GREEN}✓ huggingface-cli found in backend venv${NC}"
else
    echo -e "${YELLOW}⚠ huggingface-cli not found. Installing via uv tool...${NC}"
    uv tool install huggingface-hub
    HF_CLI="huggingface-cli"
    echo -e "${GREEN}✓ huggingface-cli installed${NC}"
fi

# 3. Download LTX-2.3 distilled MLX model (~42GB)
LTX_MODEL="notapalindrome/ltx2-mlx-av"
LTX_CACHE_DIR="$HF_CACHE/models--notapalindrome--ltx2-mlx-av"

echo ""
echo -e "${BLUE}[1/2] LTX-2.3 distilled MLX model${NC}"
echo -e "  Model: ${LTX_MODEL}"
echo -e "  Size:  ~42GB"

if [ -d "$LTX_CACHE_DIR" ] && [ -f "$LTX_CACHE_DIR/refs/main" ]; then
    echo -e "${GREEN}✓ Already downloaded${NC}"
else
    echo -e "${YELLOW}  Downloading (this may take a while)...${NC}"
    $HF_CLI download "$LTX_MODEL"
    echo -e "${GREEN}✓ LTX-2.3 model downloaded${NC}"
fi

# 4. Download Qwen3.5-2B-4bit (MLX quantized) for prompt enhancement (~1.2GB)
QWEN_MODEL="mlx-community/Qwen3.5-2B-4bit"
QWEN_CACHE_DIR="$HF_CACHE/models--mlx-community--Qwen3.5-2B-4bit"

echo ""
echo -e "${BLUE}[2/2] Qwen3.5-2B-4bit prompt enhancer (MLX quantized)${NC}"
echo -e "  Model: ${QWEN_MODEL}"
echo -e "  Size:  ~1.2GB"

if [ -d "$QWEN_CACHE_DIR" ] && [ -f "$QWEN_CACHE_DIR/refs/main" ]; then
    echo -e "${GREEN}✓ Already downloaded${NC}"
else
    echo -e "${YELLOW}  Downloading...${NC}"
    $HF_CLI download "$QWEN_MODEL"
    echo -e "${GREEN}✓ Qwen3.5-2B-4bit downloaded${NC}"
fi

# 5. Verify and report
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Download Summary                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"

TOTAL_SIZE=0

if [ -d "$LTX_CACHE_DIR" ]; then
    LTX_SIZE=$(du -sg "$LTX_CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
    echo -e "  LTX-2.3:   ${GREEN}${LTX_SIZE}GB${NC} — $LTX_CACHE_DIR"
    TOTAL_SIZE=$((TOTAL_SIZE + LTX_SIZE))
else
    echo -e "  LTX-2.3:   ${RED}not found${NC}"
fi

if [ -d "$QWEN_CACHE_DIR" ]; then
    QWEN_SIZE=$(du -sg "$QWEN_CACHE_DIR" 2>/dev/null | cut -f1 || echo "?")
    echo -e "  Qwen3.5-4bit: ${GREEN}${QWEN_SIZE}GB${NC} — $QWEN_CACHE_DIR"
    TOTAL_SIZE=$((TOTAL_SIZE + QWEN_SIZE))
else
    echo -e "  Qwen3.5:   ${RED}not found${NC}"
fi

echo -e "  Total:     ${GREEN}~${TOTAL_SIZE}GB${NC}"
echo ""
echo -e "${GREEN}✓ Model download complete!${NC}"
echo -e "  Start dev: ./scripts/dev.sh"
