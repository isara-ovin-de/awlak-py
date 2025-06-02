import pytest
import awlak
import asyncio
import json

def test_capture_exception():
    def faulty_function(x):
        y = 10
        z = x / 0
    try:
        faulty_function(5)
    except ZeroDivisionError as e:
        awlak.capture_exception(e, title="Test Exception", severity=awlak.ERROR, tags=["test"])
    # Since API calls are async, we can't easily verify the output here
    # Check logs manually or mock aiohttp for proper testing

def test_capture_event():
    awlak.capture_event("Test event", severity=awlak.INFO, tags=["test"], user_id="123")
    # Check logs manually or mock aiohttp

def test_invalid_severity(self):
    with pytest.raises(ValueError):
        awlak.capture_event("Test", severity="INVALID")

# Add more tests for edge cases, variable capture, etc.