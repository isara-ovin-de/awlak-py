# For Awlak to send captured exception or event data to an API endpoint,
# the `AWLAK_API_KEY` environment variable must be set.
# If `AWLAK_API_KEY` is not set, the captured data will be printed
# to the console by default instead of being sent to an API.

import awlak
import time 

def faulty_function(x):
    y = 10
    z = x / 0
    
try:
    faulty_function(5)
except ZeroDivisionError as e:
    awlak.capture_exception(e, title="Test", severity=awlak.ERROR, tags=["math"])
    # Only a workaround for standalone execution to prevent immediate exit
    # TODO: Remove this in production code
    time.sleep(3)
