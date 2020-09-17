"""
Date : 2018. 9. 1
Author : Jiwoo Park
"""

import time
import threading
import itertools
import sys

class Spinner():
    spinner_char = itertools.cycle(['-','/','|','\\'])
    def __init__(self):
        self.stop_running = threading.Event()
        self.spin_thread = threading.Thread(target=self.init_spin)

    def start(self):
        self.spin_thread.start()

    def stop(self):
        self.stop_running.set()
        self.spin_thread.join()

    def init_spin(self):
        while not self.stop_running.is_set():
            sys.stdout.write(next(self.spinner_char))
            sys.stdout.flush()
            time.sleep(0.25)
            sys.stdout.write('\b')

def spinner(func):
    spin = Spinner()
    def inner():
        spin.start()
        func()
        spin.stop()
    return inner

@spinner
def do_work():
    time.sleep(3)

if __name__ == "__main__":
    do_work()
