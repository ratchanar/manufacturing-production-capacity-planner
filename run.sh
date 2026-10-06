#!/bin/bash
set -e
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
else
  source .venv/bin/activate
fi
python -m streamlit run app.py
