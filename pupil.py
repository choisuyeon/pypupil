# AUTHOR : Suyeon Choi
# DATE : July 31, 2018
# Open-source lib Usage : scipy, sklearn, numpy, zmq, msgpack, affine_transformer

import threading
import math
import time, datetime
import os, sys

import numpy as np
import zmq
from msgpack import loads
from scipy import stats
import scipy.io
from sklearn.cluster import KMeans

from affine_transformer import Affine_Fit


class Pupil:
    """ API

        calibrate() :
        record() :
        disconnect() :
        _save_file() :
        _idx_lut() :
    """

    addr_localhost = '127.0.0.1'
    port_pupil_remote = '50020' # default value given by Pupil
    screen_width = 1600
    screen_height = 900
    record_duration = 10 # second
    frequency = 120 # Hz

    to_points = np.array([ [-screen_width/2, screen_height/2], [screen_width/2, screen_height/2], \
                           [screen_width/2, -screen_height/2], [-screen_width/2, -screen_height/2] ]) #, [0.0, 0.0] ])

    num_cal_points = Pupil.to_points.shape[0] # currently 4


    def __init__(self):
        # PART 1. Connection to Server

        context = zmq.Context()

        # 1-1. Open a req port to talk to pupil
        req_socket = context.socket(zmq.REQ)
        req_socket.connect( "tcp://%s:%s" %(Pupil.addr_localhost, Pupil.port_pupil_remote) )

        # 1-2. Ask for the sub port
        req_socket.send( b'SUB_PORT' )
        self.sub_port = req_socket.recv()

        # 1-3. Open a sub port to listen to gaze
        self.sub_socket = context.socket(zmq.SUB)
        # You can select topic between "gaze" and "pupil"
        data_type = b'pupil.'
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, data_type)

        if data_type == b'pupil.' :
            self.idx_left_eye = -2
        else :
            self.idx_left_eye = -3


    def calibrate(self):
        """
            Calibration order
            1st : Left top
            2nd : Right top
            3rd : Right bottom
            4th : Left bottom
            5th : Center (if needed)
        """
        # Beep sound
        sys.stdout.write('\a')
        sys.stdout.flush()

        self.sub_socket.connect(b"tcp://%s:%s" %(Pupil.addr_localhost.encode('utf-8'), self.sub_port))
        topic, msg = self.sub_socket.recv_multipart()
        pupil_position = loads(msg)
        time0 = pupil_position[b'timestamp']
        left_eye = int(str(topic)[self.idx_left_eye]) # 1 : left, 0 : right

        # Coordinate transformation
        max_num_points = 3000
        duration_per_point = 5 # second
        data = np.zeros([max_num_points * Pupil.num_cal_points, 3])

        # initialization
        t = 0
        position = 0
        index = 0
        global_idx = 0
        X = np.empty(shape = [0, 2]) # data jar
        index_position_change = []

        while t < duration_per_point * Pupil.num_cal_points :
            topic, msg = self.sub_socket.recv_multipart()
            pupil_position = loads(msg)
            x, y = pupil_position[b'norm_pos']
            t = pupil_position[b'timestamp'] - time0

            new_position = int( t / duration_per_point )
            if new_position > position :
                print('\a') # change gaze position with beep sound

                position = new_position # update position
                if position == Pupil.num_cal_points :
                    break

                index_position_change.append(global_idx)
                index = 0 # initialize index

            data[global_idx, :] = [t, x, y] # for matlab

            X = np.append(X, [[x, y]], axis = 0) # for later data processing

            index = index + 1
            global_idx = global_idx + 1
            print(t, x, y)

        # TODO : Data squeezing XX

        # get points via k-mean clustering
        kmeans = KMeans(n_clusters = Pupil.num_cal_points, random_state = 0).fit(X)
        print("centers :", kmeans.cluster_centers_)
        # TODO : plot matplot will be convenient

        # index(label) lookup table
        lut = self._idx_lut(kmeans.labels_, index_position_change)
        print("lookup table : ", lut)

        # get centers from cluster and reorder
        from_points = [ kmeans.cluster_centers_[i] for i in lut ]
        print("clustered point : ", from_points)

        # affine fitting
        self.Aff_trf = Affine_Fit(from_points, Pupil.to_points)
        print("Affrin Transform is ")
        print(self.Aff_trf.To_Str())

        # save data into .mat format
        current_time = str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
        file_name = 'eye_track_calibration_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data)


        # TEMP : save after transform data for visualization
        data_refine = np.zeros([4000, 2])
        for i in range(X.shape[0]) :
            x, y = self.Aff_trf.Transform(X[i])
            data_refine[i, :] = [x, y]

        file_name = 'eye_track_calibration_refined_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data_refine)
        # TEMP END


    def record(self):
        """ 1. receive Pupil data from device
            2. Transfrom the pupil position to gaze new_position
               With Affine transform matrix with precaculated
        """
        # check whether calibrated
        if self.Aff_trf is None :
            print("You should calibrate before record.")
            return

        self.sub_socket.connect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port) )

        # Find initial point of time.
        topic, msg = self.sub_socket.recv_multipart()
        pupil_position = loads(msg)
        time0 = pupil_position[b'timestamp']

        # Make null arrays to fill.
        max_num_points = Pupil.record_duration * Pupil.frequency * 2
        data = np.zeros([max_num_points, 6])

        # variable initialization
        t = 0
        index = 0

        # Start with Beep sound
        sys.stdout.write('\a')
        sys.stdout.flush()

        while t < Pupil.record_duration:
            topic, msg = self.sub_socket.recv_multipart()
            pupil_position = loads(msg)
            raw_point = pupil_position[b'norm_pos']

            # get real coordinate with Affine Transform
            x, y = self.Aff_trf.Transform(raw_point)

            # get time
            t = pupil_position[b'timestamp'] - time0

            ## TEST START
            ## this can be the cause of delay.
            left_eye = int(str(topic)[self.idx_left_eye]) # 1 : left, 0 : right
            ## TEST END

            print(index, topic, t, x, y)

            data[index, :] = [t, x, y, raw_point[0], raw_point[1], left_eye]
            index = index + 1

        # Beep sound
        sys.stdout.write('\a')
        sys.stdout.flush()

        # PART 3. Convert and save MATLAB file
        current_time = str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
        file_name = 'eye_track_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data)


    def disconnect(self):
        self.sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))


    def _save_file(self, file_name, data):
        """ save data in .mat format with file_name
            You can change the directory which the file will be saved.
        """

        # Assign directory
        linux_prefix = '/mnt/c'
        file_dir = linux_prefix + '/Users/User/Desktop/eye_tracker/data/'
        # file name ex : eye_track_data_180101_120847.mat
        file_name = file_dir + file_name

        scipy.io.savemat(file_name, mdict = {'data' : data})


    def _idx_lut(self, labels, index_change):
        """ get mode of labels in subarray
            and make lookup Table """
        #print("Labels.shape : ", labels.shape)
        #print("Indices : ", index_change)

        num_points = len(index_change) + 1

        spl = np.split(labels, index_change)
        LUT = np.zeros(num_points, dtype = int)

        # Find mode value
        for i in range(num_points):
            points = np.array([spl[i]]).T
            mode_index = stats.mode(points)[0][0]
            print("mode_index : ", mode_index)
            LUT[i] = mode_index[0]

        return LUT
