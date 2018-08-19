#################################################################################
# AUTHOR : Suyeon Choi
# DATE : July 27, 2018
#
# Eye tracker controller module based on the pupil_basic.py by Seokil, Moon
#################################################################################
import zmq
from msgpack import loads
import threading
import numpy as np
import math
import time
import datetime
import scipy.io

__ADDR__ = '127.0.0.1' # LOCALHOST
__REQ_PORT__ = '50020' # DEFAULT VALUE

context = zmq.Context()

# 1-1. Open a req port to talk to pupil
req_socket = context.socket(zmq.REQ)
req_socket.connect( "tcp://%s:%s" %(__ADDR__, __REQ_PORT__) )

# 1-2. Ask for the sub port
req_socket.send( b'SUB_PORT' )
sub_port = req_socket.recv()

# 1-3. Open a sub port to listen to gaze
sub_socket = context.socket(zmq.SUB)
# topic should be gaze.
sub_socket.setsockopt(zmq.SUBSCRIBE, b'gaze.')

sub_socket.connect(b"tcp://%s:%s" % (__ADDR__.encode('utf-8'), sub_port) )

# Find initial point of time.
topic, msg = sub_socket.recv_multipart()
pupil_position = loads(msg)
time0 = pupil_position[b'timestamp']

# Make null arrays to fill.
s_num = 3000
data = np.zeros([s_num, 4])

t = 0
__DURATION__ = 10 # run for 10s
index = 0

while t < __DURATION__:
    topic, msg = sub_socket.recv_multipart()
    # TODO : topic should be "gaze".

    pupil_position = loads(msg)
    x, y = pupil_position[b'norm_pos']
    t = pupil_position[b'timestamp'] - time0 # t increases here

    ## TEST START
    ## this can be the cause of delay.
    left_eye = int(str(topic)[-3]) # 1 : left, 0 : right
    ## TEST END

    print(index, topic, t, x, y)

    data[index, :] = [t, x, y, left_eye]
    index = index + 1



# PART 3. Convert and save MATLAB file
current_time = str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))

# Assigning directory
# TODO : You can change the directory which the file will be saved.
linux_prefix = '/mnt/c'
file_dir = linux_prefix + '/Users/User/Desktop/eye_tracker/data/'


file_name = file_dir + 'eye_track_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat

scipy.io.savemat(file_name, mdict = {'data' : data})

sub_socket.disconnect(b"tcp://%s:%s" % (__ADDR__.encode('utf-8'), sub_port))
