"""Training harness for distorted-camera SHARP.

This is a top-level package separate from the inference ``sharp`` package. It imports
``sharp`` from ``PYTHONPATH`` and never mutates it, so the inference path stays intact.
"""
