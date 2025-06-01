# Awlak

A lightweight Python library for monitoring exceptions and custom events with rich context in Python applications.

## Key Features

- Captures exceptions and custom events with function-local variables, code context, environment details, and tracebacks.
- Supports severity levels (`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`) for categorizing events.
- Configurable via environment variables; no manual setup required.
- Asynchronous, non-blocking API calls to avoid delaying code execution.
- Outputs to JSON (if no API key), logs to console, and optionally to a file.
- Singleton pattern ensures a single instance for consistent configuration.
- Extensible with tags and custom metadata for flexible event tracking.

## Installation

Install Awlak using pip:

```bash
pip install awlak
```

## Usage

### Capture an Exception

Capture exceptions with detailed context, including function-local variables and code snippets:

```python
import awlak

def faulty_function(x):
    y = 10
    z = x / 0
try:
    faulty_function(5)
except ZeroDivisionError as e:
    awlak.capture_exception(e, title="Division Error", severity=awlak.ERROR, tags=["math"], context="Test")
```

### Capture a Custom Event

Log custom events, such as user actions or system milestones:

```python
import awlak

awlak.capture_event("User logged in", title="User Login", severity=awlak.INFO, tags=["auth"], user_id="123")
```

## Configuration

Configure Awlak using environment variables. No manual configuration is needed. Set variables in your shell or via a `.env` file (handled externally, e.g., with `python-dotenv`).

```bash
# Example environment variables
export AWLAK_API_KEY="your-api-key"  # Required for API calls
export AWLAK_API_ENDPOINT="https://api.awlak.com/exception"  # Optional, default value
export AWLAK_LOG_FILE="/path/to/awlak.log"  # Optional, for file logging
export AWLAK_API_TIMEOUT="5"  # Optional, defaults to 5 seconds
export AWLAK_API_RETRIES="3"  # Optional, defaults to 3
export AWLAK_LOG_LEVEL="INFO"  # Optional, defaults to INFO
```

Example using a `.env` file (requires `python-dotenv`):

```plaintext
AWLAK_API_KEY=your-api-key
AWLAK_LOG_FILE=awlak.log
```

```python
from dotenv import load_dotenv
load_dotenv()
import awlak
```

## Severity Levels

- `awlak.CRITICAL`: Critical issues requiring immediate attention.
- `awlak.ERROR`: Errors impacting functionality (default for exceptions).
- `awlak.WARNING`: Potential issues that don’t halt execution.
- `awlak.INFO`: Informational events (default for events).
- `awlak.DEBUG`: Detailed debugging information.

## Example JSON Output

When no API key is set, Awlak prints JSON output like this:

```json
{
  "type": "exception",
  "title": "Division Error",
  "exception_type": "ZeroDivisionError",
  "exception_message": "division by zero",
  "severity": "error",
  "environment": {
    "python_version": "3.11.9",
    "platform": "Linux-5.15.0-73-generic-x86_64-with-glibc2.35",
    "os_name": "posix",
    "working_directory": "/path/to/dir",
    "timestamp": "2025-05-31T18:40:30.123456"
  },
  "local_variables": {
    "x": "5",
    "y": "10"
  },
  "code_context": [
    "   1 | def faulty_function(x):",
    "   2 |     y = 10",
    ">>  3 |     z = x / 0",
    "   4 |     return z"
  ],
  "traceback": ["..."],
  "caused_by": [],
  "tags": ["math"],
  "kwargs": {"context": "Test"}
}
```

## Singleton Pattern

Awlak uses a singleton pattern to ensure a single instance handles all monitoring tasks, maintaining consistent configuration across your application. Simply `import awlak` and use the module directly; no instantiation is required.

## Output Options

- **API**: If `AWLAK_API_KEY` is set, data is sent asynchronously to `AWLAK_API_ENDPOINT` without blocking execution.
- **JSON**: If no API key is provided, data is printed as JSON to stdout.
- **Logging**: Events are logged to the console (always) and to a file (if `AWLAK_LOG_FILE` is set).

## Dependencies

- `aiohttp` (&gt;=3.8.0) for asynchronous API calls.

## Development

- **Source**: GitHub Repository
- **Contributing**: Fork the repository, submit pull requests, or open issues for bugs and features.
- **Testing**: Run tests with `pytest tests/test_awlak.py` after installing `pytest`.
- **License**: MIT License (see `LICENSE` file).

## Notes

- API calls are non-blocking, ensuring your application remains responsive.
- Function-local variables are captured, excluding outer scope variables for cleaner context.
- Use environment variables for configuration to keep setup simple.

For `.env` file support, integrate `python-dotenv` in your project (not included in Awlak).