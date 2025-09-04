#!/usr/bin/env python3
"""
OptILLM - OpenAI API compatible optimizing inference proxy

This is a thin wrapper that imports and runs the main server from the optillm package.
For backwards compatibility with direct execution of optillm.py.
"""

from optillm import main

if __name__ == "__main__":
    main()
