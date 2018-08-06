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
    """ API (instructions)

        * Public :
            calibrate(eye_to_clb) :
                Select eyes to calibrate (monocular, binocular)
                It starts with beep sound
                Then gaze the position with predefined reorder ::

                    Left top -> Right top -> Right bottom -> Left bottom -> Center (if needed)

                You might hear "beep" sound for each stage.
                After collecting data, pyPupil clusters the datum and finds the average position of each gaze.
                Then it applies Affine transform with fixed point.
                Finally, calibration returns the Affine Transform Matrices. (1 or 2 matrices)

            record() :


            disconnect() :
                Disconnect from the SUBSRIBE socket.


        * Private:
            _save_file(file_name, data) :
                Save numpy 2d array to .mat file wtih designated file_name.

            _idx_lut() :
                Returns Lookup Table of idx of n_clusters
                This is necessary because K-mean clustering does not guarantee
                the order of centers which I intended.

                This method get mode of labels in subarray then make lookuptable.
    """

    addr_localhost = '127.0.0.1'
    port_pupil_remote = '9285' # default value given by Pupil : 50020
                               # You should check Pupil remote tab if communication performs not well
    screen_width = 4.0
    screen_height = 2.0
    record_duration = 15 # second
    frequency = 120 # Hz

    to_points = np.array([ [-screen_width/2, screen_height/2], [screen_width/2, screen_height/2], \
                           [screen_width/2, -screen_height/2], [-screen_width/2, -screen_height/2], [0.0, 0.0] ])

    num_cal_points = to_points.shape[0] # currently 4


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
        #self.set_data_type(b'pupil.')
        print("Automatically set to deal with pupil data. (not gaze)")
        type = b'pupil.'

        self.sub_socket.setsockopt(zmq.SUBSCRIBE, type)

        if type == b'pupil.' :
            self.idx_left_eye = -2
        else :
            self.idx_left_eye = -3

        self.Affine_Transforms = [None, None]

    def calibrate(self, eye_to_clb):
        """
            Argument :
                eye_to_clb : list of integer ( only right eye : [0]
                                               only left eye : [1]
                                               both eyes : [0, 1] )

            Calibration order :
                1st : Left top
                2nd : Right top
                3rd : Right bottom
                4th : Left bottom
                5th : Center (if needed)
        """
        # Beep sound
        print('\a')
        self.sub_socket.connect(b"tcp://%s:%s" %(Pupil.addr_localhost.encode('utf-8'), self.sub_port))
        topic, msg = self.sub_socket.recv_multipart()

        # get data
        pupil_position = loads(msg)
        time0 = pupil_position[b'timestamp']
        left_eye = int(str(topic)[self.idx_left_eye]) # 1 : left, 0 : right

        # Coordinate transformation
        duration_per_point = 5 # second
        max_num_points = duration_per_point * Pupil.frequency * Pupil.num_cal_points * len(eye_to_clb)
        data_calibration = np.zeros([max_num_points * Pupil.num_cal_points, 4])

        # initialization
        t = 0
        position = 0
        index = 0
        global_idx = 0
        X = [np.empty(shape = [0, 2]), np.empty(shape = [0, 2])] # data for left and right eye
        indices_eye_position_change = [[],[]]

        while t < duration_per_point * Pupil.num_cal_points :
            topic, msg = self.sub_socket.recv_multipart()
            pupil_position = loads(msg)
            x, y = pupil_position[b'norm_pos']
            t = pupil_position[b'timestamp'] - time0
            left_eye = int(str(topic)[self.idx_left_eye])

            new_position = int( t / duration_per_point )
            if new_position > position :
                print('\a') # change gaze position with beep sound

                position = new_position # update position
                if position == Pupil.num_cal_points :
                    break

                for eye in eye_to_clb:
                    indices_eye_position_change[eye].append(len(X[eye]))
                index = 0 # initialize index

            data_calibration[global_idx, :] = [t, x, y, left_eye] # for matlab

            X[left_eye] = np.append(X[left_eye], [[x, y]], axis = 0) # for later data processing

            index = index + 1
            global_idx = global_idx + 1
            print(topic, t, x, y)

        # TODO : Data squeezing XX

        # TEMP You can delete here
        #clusters = []
        #luts = []
        # TEMP END
        from_points = [[], []]

        # processing eye by eye
        for eye in eye_to_clb:
            # (1) get points via k-mean clustering
            cluster = KMeans(n_clusters = Pupil.num_cal_points, random_state = 0).fit(X[eye])
            # TODO : plot matplot will be convenient

            # (2) index(label) lookup table
            lut = self._idx_lut(cluster.labels_, indices_eye_position_change[eye])
            #if lut == None:
                # Not clusted well
            #    print("Terminated because not clusted well.. Try again")
            #    return
            print("lookup table : ", lut)

            # (3) get centers from cluster and reorder
            from_points[eye] = [ cluster.cluster_centers_[i] for i in lut ]
            print("clustered point : ", from_points[eye])

            # (4) affine fitting (calibration)
            self.Affine_Transforms[eye] = Affine_Fit(from_points[eye], Pupil.to_points)
            if self.Affine_Transforms[eye] == False:
                print("Not clustered well, Try again")
                return

            print("Affine Transform is ")
            print(self.Affine_Transforms[eye].To_Str())

            # TEMP You can delete here
            #clusters.append(cluster)
            #luts.append(lut)
            print("centers of eye" + str(eye) + ": ", cluster.cluster_centers_)
            # TEMP END


        # save data into .mat format
        current_time = str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
        file_name = 'eye_track_before_calib_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data_calibration)
        self._save_file('eye_track_before_calib_data_latest.mat', data_calibration)

        # TEMP : save after transform data for visualization
        data_refine = np.zeros([max_num_points * Pupil.num_cal_points, 4])

        for i in range(data_calibration.shape[0]):
            eye = int(data_calibration[i][3])
            if self.Affine_Transforms[eye] is None:
                continue

            raw_data = (data_calibration[i][1], data_calibration[i][2])
            x, y = self.Affine_Transforms[eye].Transform(raw_data)
            t = data_calibration[i][0]
            data_refine[i, :] = [t, x, y, eye]

        file_name = 'eye_track_after_calib_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data_refine)
        self._save_file('eye_track_after_calib_data_latest.mat', data_refine)
        # TEMP END
        self.sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))


    def record(self):
        """ 1. receive Pupil data from device
            2. Transfrom the pupil position to gaze new_position
               With Affine transform matrix with precaculated
        """
        # check whether calibrated
        if any(self.Affine_Transforms) is False :
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
        print('\a')

        while t < Pupil.record_duration:
            topic, msg = self.sub_socket.recv_multipart()
            pupil_position = loads(msg)
            raw_point = pupil_position[b'norm_pos']
            left_eye = int(str(topic)[self.idx_left_eye]) # 1 : left, 0 : right

            # get real coordinate with Affine Transform
            x, y = self.Affine_Transforms[left_eye].Transform(raw_point)

            # get time
            t = pupil_position[b'timestamp'] - time0

            print(index, left_eye, t, x, y)

            data[index, :] = [t, x, y, left_eye, raw_point[0], raw_point[1]]
            index = index + 1

        # Beep sound
        print('\a')

        # PART 3. Convert and save MATLAB file
        current_time = str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
        file_name = 'eye_track_gaze_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data)
        self._save_file('eye_track_gaze_data_latest.mat', data)
        self.sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))

    def disconnect(self):
        self.sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))


    def _plot_graph(self, data = None):
        data = np.zeros(shape = [10, 2])

        for i in range(10):
            x, y = (i, 2*i)
            data[i, :] = [x, y]
        # Change the line plot below to a scatter plot
        print(data)
        plt.scatter(data[:, 0], data[:, 1])
        # Show plot
        plt.show()

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
        repetition_check = []

        # Find mode value
        for i in range(num_points):
            points = np.array([spl[i]]).T
            mode_index = stats.mode(points)[0][0]

            #if mode_index in repetition_check:
                # repetition check
            #    print("mode index : ", mode_index)
            #    print(str(repetition_check))
            #    return None
            #else:
            #    repetition_check.append(mode_index)
            LUT[i] = mode_index[0]

        return LUT
