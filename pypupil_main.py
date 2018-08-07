# AUTHOR : Suyeon Choi
# DATE : August 1, 2018

# Command line interface for pypupil
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
        eyes = []
        cmd_eye = input('=' * 60 +
                        '\nSelect eyes to calibrate \n' +
                        '\t\t l (left) \n' +
                        '\t\t r (right) \n' +
                        '\t\t b (binocular) \n' +
                        '=' * 60 +
                        '\nInput command : ')
        if cmd_eye == "l" or cmd_eye == "left":
            eyes.append(1)
        elif cmd_eye == "r" or cmd_eye == "right":
            eyes.append(0)
        elif cmd_eye == "b" or cmd_eye == "binocular":
            eyes.append(0)
            eyes.append(1)
        else:
            continue

        tracker.calibrate(eyes)

    elif command == "g" or command == "get_data":
        sync = False
        cmd_sync = input('=' * 60 +
                        '\nDo you want to synchronize? \n' +
                        '\t\t y (yes) \n' +
                        '\t\t n (no) \n' +
                        '=' * 60 +
                        '\nInput command : ')
        if cmd_sync == "y" or cmd_sync == "yes":
            sync = True
        elif cmd_sync == "n" or cmd_sync == "no":
            pass
        else:
            continue
        tracker.record(sync)

    elif command == "exit":
        sys.exit(1)

    elif command == "p":
        tracker._plot_graph()
