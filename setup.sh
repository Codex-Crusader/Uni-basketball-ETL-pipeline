#!/bin/bash
# ============================================================
#  NCAA Basketball Predictor — Mac/Linux Setup Script
#  Run this once after downloading or cloning the project:
#      chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo ""
echo " NCAA Basketball Predictor v2.5.1 — Setup"
echo " =========================================="
echo ""

# Check Python 3 is available
if ! command -v python3 &>/dev/null; then
    echo " ERROR: python3 not found."
    echo " Install Python 3.8+ via your package manager:"
    echo "   macOS:  brew install python3"
    echo "   Ubuntu: sudo apt install python3 python3-venv"
    exit 1
fi

echo " Python found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't already exist
if [ ! -d ".venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv .venv
    echo " Virtual environment created."
else
    echo " Virtual environment already exists, skipping creation."
fi

echo ""
echo " Activating virtual environment..."
source .venv/bin/activate

echo ""
echo " Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo ""
echo " ============================================================"
echo "  Setup complete."
echo " ============================================================"
echo ""
echo " To use the predictor, activate the environment first:"
echo ""
echo "     source .venv/bin/activate"
echo ""
echo " Then run one of these commands:"
echo ""
echo "     python main.py --fetch                Fetch ~2900 real NCAA games"
echo "     python main.py --generate-synthetic   Use offline synthetic data"
echo "     python main.py --train                Train all models"
echo "     python main.py --serve                Start dashboard at localhost:5000"
echo ""
echo " Quick start (no internet required):"
echo ""
echo "     python main.py --generate-synthetic"
echo "     python main.py --train"
echo "     python main.py --serve"
echo ""