# AUTHOR : Suyeon Choi
# DATE : August 1, 2018

import time
import sys
from pupil import Pupil

if __name__ == "__main__" :
    print("start eye tracker")

tracker = Pupil();

while True:
    time.sleep(1)
    command = input('=' * 60 +
                    '\nPossible commands \n' +
                    '\t\t c (calibrate) \n' +
                    '\t\t g (get_data) \n' +
                    '\t\t exit \n' +
                    '=' * 60 +
                    '\nInput command : ')
    print('\n')
    if command == "c" or command == "calibrate":
        tracker.calibrate()

    elif command == "g" or command == "get_data":
        tracker.record()

    elif command == "exit":
        tracker.disconnect()
        sys.exit(1)
