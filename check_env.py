import sys
import os

print(f"1. Python is running from: {sys.executable}")
print(f"   (Should contain 'bc_ai')")

try:
    import pandas
    print(f"2. Pandas version: {pandas.__version__}")
    print("   SUCCESS: Pandas is found!")
except ImportError:
    print("   ERROR: Pandas is MISSING. You are in the wrong environment.")