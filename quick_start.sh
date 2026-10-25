#!/bin/bash
# Quick Start Script for RF-DETR Logo Detection
# Run this after Phase 1 completion to verify setup

echo "=================================================="
echo "RF-DETR Logo Detection - Quick Start"
echo "=================================================="
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check GPU
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Run verification test
echo ""
echo "Running installation verification..."
python test_setup.py

echo ""
echo "=================================================="
echo "Next Steps:"
echo "=================================================="
echo "1. Get Roboflow API key from https://app.roboflow.com/settings/api"
echo "2. Copy .env.example to .env and add your key"
echo "3. Run dataset download: python scripts/download_dataset.py"
echo ""
echo "For detailed instructions, see SETUP.md"
echo "=================================================="
