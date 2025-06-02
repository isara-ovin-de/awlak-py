import awlak

def faulty_function(x):
    y = 10
    z = x / 0
    
try:
    faulty_function(5)
except ZeroDivisionError as e:
    awlak.capture_exception(e, title="Test", severity=awlak.ERROR, tags=["math"])