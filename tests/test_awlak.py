import pytest
import awlak
import asyncio
import json
import os
from unittest.mock import patch, AsyncMock

# Configure logging to be quiet during tests
import logging
logging.getLogger("awlak").setLevel(logging.CRITICAL)


def test_capture_exception_no_api_key(): # Renamed for clarity
    # Test that when no API key is set, console output happens (implicitly)
    # and API is not called.
    original_instance = awlak._instance
    awlak._instance = None # Force re-init

    original_api_key = os.environ.get("AWLAK_API_KEY")
    if "AWLAK_API_KEY" in os.environ:
        del os.environ["AWLAK_API_KEY"]

    # Patch _send_to_api for this specific test
    with patch('awlak.Awlak._send_to_api', new_callable=AsyncMock) as mock_no_api_call:
        client = awlak.Awlak() # Re-initializes with no API key due to env var absence

        def faulty_function(x):
            y = 10
            z = x / 0
        try:
            faulty_function(5)
        except ZeroDivisionError as e:
            client.capture_exception(e, title="Test Exception No Key", severity=awlak.ERROR, tags=["test"])

        mock_no_api_call.assert_not_called()

    # Teardown
    awlak._instance = original_instance
    if original_api_key is not None:
        os.environ["AWLAK_API_KEY"] = original_api_key
    # No need to clean up client._loop as it shouldn't create one if no API key


def test_capture_event_no_api_key(): # Renamed for clarity
    original_instance = awlak._instance
    awlak._instance = None # Force re-init

    original_api_key = os.environ.get("AWLAK_API_KEY")
    if "AWLAK_API_KEY" in os.environ:
        del os.environ["AWLAK_API_KEY"]

    with patch('awlak.Awlak._send_to_api', new_callable=AsyncMock) as mock_no_api_call_event:
        client = awlak.Awlak() # Re-initializes

        client.capture_event("Test event no key", severity=awlak.INFO, tags=["test"], user_id="123")
        mock_no_api_call_event.assert_not_called()

    # Teardown
    awlak._instance = original_instance
    if original_api_key is not None:
        os.environ["AWLAK_API_KEY"] = original_api_key


def test_capture_exception(): # Original test_capture_exception name
    # This is the old test, let's assume it was for manual log checking without API key
    # For now, it's duplicative of test_capture_exception_no_api_key if we mock.
    # Keeping it simple for now, or it could be removed if covered.
    # To avoid issues, let's ensure it also has the instance reset logic if it calls awlak.
    original_instance = awlak._instance
    awlak._instance = None
    original_api_key = os.environ.get("AWLAK_API_KEY")
    if "AWLAK_API_KEY" in os.environ:
        del os.environ["AWLAK_API_KEY"]

    # awlak.Awlak() # Ensure instance is created if not already by capture_exception

    def faulty_function(x):
        y = 10
        z = x / 0
    try:
        faulty_function(5)
    except ZeroDivisionError as e:
        awlak.capture_exception(e, title="Test Exception", severity=awlak.ERROR, tags=["test"])
    # Since API calls are async, we can't easily verify the output here
    # Check logs manually or mock aiohttp for proper testing
    # Teardown
    awlak._instance = original_instance
    if original_api_key is not None:
        os.environ["AWLAK_API_KEY"] = original_api_key


def test_capture_event(): # Original test_capture_event
    # Similar to above, this is likely for manual checking without API key
    original_instance = awlak._instance
    awlak._instance = None
    original_api_key = os.environ.get("AWLAK_API_KEY")
    if "AWLAK_API_KEY" in os.environ:
        del os.environ["AWLAK_API_KEY"]

    # awlak.Awlak()

    awlak.capture_event("Test event", severity=awlak.INFO, tags=["test"], user_id="123")
    # Check logs manually or mock aiohttp
    # Teardown
    awlak._instance = original_instance
    if original_api_key is not None:
        os.environ["AWLAK_API_KEY"] = original_api_key


def test_invalid_severity(): # Removed self
    # Ensure instance is managed if using global awlak.capture_event
    original_instance = awlak._instance
    awlak._instance = None
    client = awlak.Awlak() # Use a local client for this test to avoid global state issues

    with pytest.raises(ValueError):
        client.capture_event("Test", severity="INVALID")

    awlak._instance = original_instance


@patch('awlak.Awlak._send_to_api', new_callable=AsyncMock)
async def test_capture_exception_sends_to_api(mock_send_to_api: AsyncMock):
    original_awlak_instance = awlak._instance
    awlak._instance = None # Force re-initialization of the singleton

    original_api_key = os.environ.get("AWLAK_API_KEY")
    os.environ["AWLAK_API_KEY"] = "test_key_for_api_send"

    # client needs to be initialized after env var is set and instance is cleared
    # and _send_to_api is patched for this instance of Awlak
    client = awlak.Awlak()

    # Ensure that the mock is associated with the client instance
    # This is tricky with module-level patch if Awlak instance is not fresh
    # The safest is to patch on the class `awlak.Awlak` then instantiate.
    # The patch decorator should handle this correctly by patching Awlak._send_to_api
    # before `client = awlak.Awlak()` is called.

    def faulty_function(x):
        y = 10
        return x / 0

    try:
        faulty_function(5)
    except ZeroDivisionError as e:
        client.capture_exception(e, title="Test API Send", severity=awlak.ERROR)

    # Wait for the task to be processed
    if client._own_loop:
        assert len(client._pending_api_calls) >= 1, "No pending API calls found when expected."
        # Get the last submitted future, assuming it's the one from this test
        future = client._pending_api_calls[-1]
        try:
            await asyncio.wait_for(future, timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("_send_to_api call (via run_coroutine_threadsafe) timed out.")
        except Exception as exc_fut:
            pytest.fail(f"_send_to_api call (via run_coroutine_threadsafe) failed with {exc_fut}")
    else:
        # If using an existing loop (e.g. pytest-asyncio's loop),
        # asyncio.create_task was used. Yield control to allow it to run.
        await asyncio.sleep(0.1) # Increased sleep to be safer

    mock_send_to_api.assert_called_once()
    call_args = mock_send_to_api.call_args[0][0] # data is the first arg
    assert call_args['title'] == "Test API Send"
    assert call_args['type'] == "exception"

    # Teardown for this test
    if client._own_loop and client._thread and client._thread.is_alive():
        # Signal the loop to stop and join the thread
        # This is important to clean up resources if Awlak started its own thread
        client._loop.call_soon_threadsafe(client._loop.stop)
        client._thread.join(timeout=2.0)
        if client._thread.is_alive():
            print("Warning: Awlak thread did not exit cleanly after test_capture_exception_sends_to_api.")
        if not client._loop.is_closed():
            # This call is not thread-safe and should be done from within the loop's thread or after it has stopped
            # client._loop.close()
            pass


    awlak._instance = original_awlak_instance # Restore original instance
    if original_api_key is not None:
        os.environ["AWLAK_API_KEY"] = original_api_key
    elif "AWLAK_API_KEY" in os.environ: # If it was set by this test but not originally
        del os.environ["AWLAK_API_KEY"]


# Add more tests for edge cases, variable capture, etc.