"""Test configuration and fixtures.

Ensures JAX is configured with x64 support enabled for full float64 precision
during tests, matching AMSA's float64 default.
"""
import os

# Enable JAX x64 mode before importing JAX
os.environ["JAX_ENABLE_X64"] = "1"
