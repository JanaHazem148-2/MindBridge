#!/bin/bash
# MindBridge Startup Script
# ─────────────────────────────────────────────────────────────
# Usage: bash start.sh
# Or:    chmod +x start.sh && ./start.sh
# ─────────────────────────────────────────────────────────────

set -e

echo ""
echo "  🧠  MindBridge — Starting up..."
echo "  ────────────────────────────────"
echo ""

# Check .env
if [ ! -f .env ]; then
  echo "  ⚠️  .env not found. Creating from .env.example..."
  cp .env.example .env
  echo "  → Edit .env and add your GROQ_API_KEY, then re-run."
  echo ""
fi

# Check for GROQ_API_KEY
source .env 2>/dev/null || true
if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "gsk_PASTE_YOUR_KEY_HERE" ]; then
  echo "  ❌  GROQ_API_KEY not set in .env"
  echo "  → Get a free key at: https://console.groq.com/keys"
  echo "  → Add it to .env:  GROQ_API_KEY=gsk_..."
  echo ""
  exit 1
fi

# Install deps
echo "  📦  Installing dependencies..."
pip install flask flask-cors groq --quiet --break-system-packages 2>/dev/null || \
pip install flask flask-cors groq --quiet 2>/dev/null || true

echo ""
echo "  ✅  Starting MindBridge on http://localhost:5000"
echo ""
echo "  Open your browser at: http://localhost:5000"
echo "  Press Ctrl+C to stop"
echo ""

# Run
python server.py
