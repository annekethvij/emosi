#!/usr/bin/env python
"""
Wrapper script to run the EMOSI system.
"""

import os
import sys

# Add the emosi directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emosi'))

# Import and run the CLI demo
from main import run_cli_demo

if __name__ == "__main__":
    run_cli_demo() 