import sys
import os

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        if os.path.exists(filename):
            os.remove(filename)
        self.log = open(filename, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        # This method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some behavior here.
        pass