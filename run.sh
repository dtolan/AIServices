#!/bin/bash

echo "Starting SD Prompt Assistant..."
echo ""

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Python not found! Please install Python 3.9 or higher."
    exit 1
fi

echo ""
echo "Starting FastAPI server..."
python3 -m backend.main
