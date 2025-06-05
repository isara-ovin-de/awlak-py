import sys
import traceback
import inspect
import os
import platform
import datetime
import json
import logging
import asyncio
import threading
import atexit
from typing import Any, Dict, Optional, List, Union
import aiohttp
from aiohttp import ClientSession

class Awlak:
    # Severity levels
    CRITICAL: str = "critical"
    ERROR: str = "error"
    WARNING: str = "warning"
    INFO: str = "info"
    DEBUG: str = "debug"
    _VALID_SEVERITIES: set = {CRITICAL, ERROR, WARNING, INFO, DEBUG}

    _instance = None

    def __new__(cls):
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization if already instantiated
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Configuration from environment variables
        self.api_endpoint: str = os.environ.get("AWLAK_API_ENDPOINT", "https://api.awlak.com/exception")
        self.api_key: Optional[str] = os.environ.get("AWLAK_API_KEY")
        self.api_timeout: int = int(os.environ.get("AWLAK_API_TIMEOUT", 5))
        self.api_retries: int = int(os.environ.get("AWLAK_API_RETRIES", 3))
        self.log_file: Optional[str] = os.environ.get("AWLAK_LOG_FILE")
        self.log_level: str = os.environ.get("AWLAK_LOG_LEVEL", "INFO")

        self._setup_logging()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._own_loop: bool = False
        self._pending_api_calls: List[asyncio.Future] = []

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError: # No running loop in current thread
            pass # self._loop remains None

        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._own_loop = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            atexit.register(self._shutdown_loop)
            self.logger.info("Awlak initialized with its own event loop in a separate thread.")
        else:
            self._own_loop = False
            self.logger.info("Awlak initialized using an existing event loop in the current thread.")


    def _run_loop(self) -> None:
        """Runs the asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _shutdown_loop(self) -> None:
        """Gracefully shuts down the event loop and thread if Awlak created them."""
        if not self._own_loop or not self._loop or not self._thread:
            return

        self.logger.info("Awlak shutting down its event loop...")

        # Wait for pending API calls
        # Make a copy for safe iteration as futures might be removed if they complete quickly
        for future in list(self._pending_api_calls):
            if not future.done():
                try:
                    # Wait for the future to complete with a timeout
                    future.result(timeout=self.api_timeout + 1) # Add a bit more timeout than API
                except asyncio.TimeoutError:
                    self.logger.warning(f"A task did not complete in time during shutdown: {future}")
                except Exception as e:
                    self.logger.error(f"Exception while waiting for task during shutdown: {e}")
            # Remove from list if present (it might have been removed by the task itself upon completion too)
            if future in self._pending_api_calls:
                self._pending_api_calls.remove(future)


        if self._loop.is_running():
            self.logger.info("Stopping event loop...")
            self._loop.call_soon_threadsafe(self._loop.stop)

        self.logger.info("Joining event loop thread...")
        self._thread.join(timeout=self.api_timeout + 2) # Wait for thread to finish
        if self._thread.is_alive():
            self.logger.warning("Event loop thread did not join cleanly.")

        # Final close of the loop
        if not self._loop.is_closed():
             # Ensure all tasks are cancelled before closing, run_until_complete for pending tasks
            try:
                # Gather all remaining tasks (if any)
                remaining_tasks = asyncio.all_tasks(loop=self._loop)
                if remaining_tasks:
                    self.logger.info(f"Cancelling {len(remaining_tasks)} remaining tasks before closing loop.")
                    for task in remaining_tasks:
                        task.cancel()
                    # Run loop until all tasks are cancelled
                    # self._loop.run_until_complete(asyncio.gather(*remaining_tasks, return_exceptions=True))
                    # We need to be careful here, run_until_complete should not be called from a different thread
                    # if the loop is already stopped. call_soon_threadsafe might be a better primitive.
                    # For now, let's rely on the loop.stop() and thread.join().
                    # A more robust solution might involve a more complex shutdown sequence within the loop's thread.
                    pass # For now, rely on stop() and join()
            except Exception as e:
                self.logger.error(f"Error during final task cancellation: {e}")
            finally:
                 self._loop.close()

        self.logger.info("Awlak shutdown complete.")


    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        self.logger = logging.getLogger("awlak")
        self.logger.setLevel(self.log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if configured
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _get_environment(self) -> Dict[str, str]:
        """Capture environment details."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "os_name": os.name,
            "working_directory": os.getcwd(),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def _get_code_context(self, frame, lines_before: int = 3, lines_after: int = 3) -> List[str]:
        """Get the lines of code around the error or event."""
        try:
            source_lines, start_line = inspect.getsourcelines(frame)
            error_line = frame.f_lineno - start_line
            start = max(0, error_line - lines_before)
            end = min(len(source_lines), error_line + lines_after + 1)
            context = []
            for i in range(start, end):
                line_num = start_line + i + 1
                prefix = ">> " if i == error_line else "   "
                context.append(f"{prefix}{line_num:4d} | {source_lines[i].rstrip()}")
            return context
        except Exception:
            return ["Could not retrieve code context"]

    def _get_local_variables(self, frame) -> Dict[str, str]:
        """Capture local variables defined in the function's scope."""
        def safe_repr(value):
            try:
                rep = repr(value)
                return rep[:1000] + "..." if len(rep) > 1000 else rep
            except Exception:
                return "<unserializable>"

        try:
            # Get the code object of the function
            code_obj = frame.f_code
            # Get names of local variables and parameters
            local_var_names = code_obj.co_varnames[:code_obj.co_argcount + code_obj.co_kwonlyargcount + code_obj.co_nlocals]
            # Filter f_locals to include only variables defined in the function
            local_vars = {
                key: safe_repr(value)
                for key, value in frame.f_locals.items()
                if key in local_var_names
            }
            return local_vars
        except Exception:
            return {"error": "<could not retrieve local variables>"}

    async def _send_to_api(self, data: Dict[str, Any]) -> bool:
        """Send data to the API asynchronously if AWLAK_API_KEY is set."""
        if not self.api_key:
            self.logger.warning("No AWLAK_API_KEY set, skipping API call")
            return False

        event_type = data.get("type", "unknown_type")
        event_title = data.get("title", "untitled_event")
        self.logger.info(f"Starting API call process for {event_type} '{event_title}'...")

        async def attempt_request(session: ClientSession, retries: int) -> bool:
            for attempt in range(retries + 1):
                try:
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    async with session.post(
                        self.api_endpoint,
                        json=data,
                        headers=headers,
                        timeout=self.api_timeout
                    ) as response:
                        if response.status == 200:
                            self.logger.info("Successfully sent data to API")
                            return True
                        else:
                            self.logger.error(f"API call failed with status {response.status}")
                except aiohttp.ClientError as e:
                    self.logger.error(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, etc.
                    await asyncio.sleep(0.1 * (2 ** attempt))
            self.logger.error(f"API call failed for {event_type} '{event_title}' after {retries + 1} attempts.")
            return False

        async with aiohttp.ClientSession() as session:
            return await attempt_request(session, self.api_retries)

    def _format_output(self, data: Dict[str, Any]) -> str:
        """Format data as JSON for printing."""
        return json.dumps(data, indent=2)

    def capture_exception(self, exception: Exception, severity: str = ERROR, **kwargs) -> None:
        """Capture and process an exception.

        Args:
            exception: The exception to capture.
            severity: Severity level (e.g., awlak.ERROR, awlak.CRITICAL).
            **kwargs: Additional metadata (e.g., title, tags).
        """
        if severity not in self._VALID_SEVERITIES:
            raise ValueError(f"Invalid severity. Must be one of {self._VALID_SEVERITIES}")

        # Get exception info
        exc_type, exc_value, exc_traceback = sys.exc_info() if isinstance(exception, Exception) else (type(exception), exception, exception.__traceback__)

        # Find the frame where the exception was raised
        frame = None
        if exc_traceback:
            # Traverse to the last frame in the traceback (where the exception originated)
            tb = exc_traceback
            while tb.tb_next:
                tb = tb.tb_next
            frame = tb.tb_frame

        if frame is None:
            # Fallback to caller's frame if no traceback (should be rare)
            frame = inspect.currentframe().f_back

        # Handle chained exceptions
        caused_by = []
        current_exc = exc_value
        while current_exc and current_exc.__cause__:
            current_exc = current_exc.__cause__
            caused_by.append({
                "exception_type": type(current_exc).__name__,
                "exception_message": str(current_exc),
            })

        data = {
            "type": "exception",
            "title": kwargs.get("title", f"{exc_type.__name__}: {str(exc_value)}"),
            "exception_type": exc_type.__name__,
            "exception_message": str(exc_value),
            "severity": severity,
            "environment": self._get_environment(),
            "local_variables": self._get_local_variables(frame),
            "code_context": self._get_code_context(frame),
            "traceback": traceback.format_tb(exc_traceback) if exc_traceback else [],
            "caused_by": caused_by,
            "tags": kwargs.get("tags", []),
            "kwargs": {k: v for k, v in kwargs.items() if k not in ("title", "tags", "severity")},
        }

        self.logger.error(f"Captured exception: {data['title']}")
        if not self.api_key:
            print(self._format_output(data))
        else:
            if self._own_loop:
                future = asyncio.run_coroutine_threadsafe(self._send_to_api(data), self._loop)
                self._pending_api_calls.append(future)
            else:
                # If using an external loop, assume it's managed elsewhere
                asyncio.create_task(self._send_to_api(data))

    def capture_event(self, event: Any, severity: str = INFO, **kwargs) -> None:
        """Capture and process a custom event.

        Args:
            event: The event to capture (e.g., string, object).
            severity: Severity level (e.g., awlak.INFO, awlak.DEBUG).
            **kwargs: Additional metadata (e.g., title, tags).
        """
        if severity not in self._VALID_SEVERITIES:
            raise ValueError(f"Invalid severity. Must be one of {self._VALID_SEVERITIES}")

        frame = inspect.currentframe().f_back

        data = {
            "type": "event",
            "title": kwargs.get("title", f"Event: {str(event)}"),
            "event_description": str(event),
            "severity": severity,
            "environment": self._get_environment(),
            "local_variables": self._get_local_variables(frame),
            "code_context": self._get_code_context(frame),
            "tags": kwargs.get("tags", []),
            "kwargs": {k: v for k, v in kwargs.items() if k not in ("title", "tags", "severity")},
        }

        self.logger.info(f"Captured event: {data['title']}")
        if not self.api_key:
            print(self._format_output(data))
        else:
            if self._own_loop:
                future = asyncio.run_coroutine_threadsafe(self._send_to_api(data), self._loop)
                self._pending_api_calls.append(future)
            else:
                # If using an external loop, assume it's managed elsewhere
                asyncio.create_task(self._send_to_api(data))

# Singleton instance
_instance = Awlak()

# Expose methods at module level
capture_exception = _instance.capture_exception
capture_event = _instance.capture_event

# Expose severity constants at module level
CRITICAL = _instance.CRITICAL
ERROR = _instance.ERROR
WARNING = _instance.WARNING
INFO = _instance.INFO
DEBUG = _instance.DEBUG

# Example usage
if __name__ == "__main__":
    def faulty_function(x):
        y = 10
        z = x / 0  # This will raise a ZeroDivisionError
        return z

    # Example with manual exception capture
    try:
        faulty_function(5)
    except ZeroDivisionError as e:
        capture_exception(e, title="Division Error", severity=ERROR, tags=["math"], context="Testing")

    # Example with event capture
    capture_event("User logged in", title="User Event", severity=INFO, tags=["auth"], user_id=123)

    # Example with warning event
    capture_event("Deprecated function used", title="Deprecation Warning", severity=WARNING, tags=["deprecation"])