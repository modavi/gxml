""" Provides a way to measure performance of code blocks. """

import inspect

eventStack = []

PROFILE_ENABLED = False

def push_perf_marker(name = None):
    if not PROFILE_ENABLED:
        return
    
    try:
        stack = inspect.stack()
        name = name or stack[1].function
         
        event = __import__("hou").perfMon.startEvent(name)
        eventStack.append(event)
        return event
    except Exception as e:
        print("EXCEPTION " + str(e))
        pass
    
def pop_perf_marker():
    if not PROFILE_ENABLED:
        return
    
    try:
        if len(eventStack) == 0:
            raise Exception("No events in the stack to stop")
        
        event = eventStack.pop()
        
        if event:
            event.stop()
    except Exception as e:
        print("EXCEPTION " + str(e))
        pass