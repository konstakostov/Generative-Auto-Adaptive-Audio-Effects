import keyboard
import time

def wait_for_spacebar():
    """
    Wait for the spacebar press and return.
    """
    keyboard.wait("space")

    # Small delay to avoid double-triggers
    time.sleep(0.1)
