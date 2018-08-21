"""
.. author:: Suyeon Choi <0310csy@hanmail.net>

"""

import threading
import math
import time, datetime
import os, sys
from queue import Queue # python 3.7

import numpy as np
import zmq
from msgpack import loads
from scipy import stats
import scipy.io
from sklearn.cluster import KMeans

from affine_transformer import Affine_Fit


class Pupil:
    """
        Pupil-labs Eye tracker controller module in Python3

        July 31, 2018 Suyeon Choi

    """

    addr_localhost = '127.0.0.1'
    port_pupil_remote = '50020' # default value given by Pupil : 50020
                                # You should check Pupil remote tab when communication is not good.

    frequency = 120 # Hz
    period = 1 / frequency # second







    def __init__(self, port_remote = '50020'):
        # 1. Connection to Server
        context = zmq.Context()

        # 1-1. Open a req port to talk to pupil
        req_socket = context.socket(zmq.REQ)
        Pupil.port_pupil_remote = port_remote
        req_socket.connect( "tcp://%s:%s" %(Pupil.addr_localhost, Pupil.port_pupil_remote) )

        # 1-2. Ask for the sub port
        req_socket.send( b'SUB_PORT' )
        self.sub_port = req_socket.recv()

        # 1-3. Open a sub port to listen to gaze
        self.sub_socket = context.socket(zmq.SUB)

        # You can select topic between "gaze" and "pupil"
        print("Automatically set to deal with pupil data. (not gaze)")
        topic = b'pupil.'
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, topic)
        if topic == b'pupil.' :
            self.idx_left_eye = -2
        else :
            self.idx_left_eye = -3

        self.Affine_Transforms = [None, None]



        self.duration_calibrate = 6 # second
        self.duration_record = 20 # second
        self.dummy_period = 1 # second
        self.screen_width = 54.0
        self.screen_height = 32.0
        self.to_points = np.array([ [-self.screen_width/2, self.screen_height/2], [self.screen_width/2, self.screen_height/2], \
                               [self.screen_width/2, -self.screen_height/2], [-self.screen_width/2, -self.screen_height/2], [0.0, 0.0] ])
        self.num_cal_points = self.to_points.shape[0] # currently 5


    """
    Public Methods
    """
    def calibrate(self, eye_to_clb = [0, 1], USE_DUMMY_PERIOD = True):
        '''
        :param list eye_to_clb:
                                list of integer ( only right eye : [0]
                                                  only left eye : [1]
                                                  both eyes : [0, 1] )

        :param bool USE_DUMMY_PERIOD: True if you want to discard the data in transition of gaze.


        Select eyes to calibrate (monocular, binocular)
        It starts with beep sound
        Then gaze the position with predefined reorder ::

            Left top -> Right top -> Right bottom -> Left bottom -> Center (if needed)

            1 ━━━━━━━━━━━━━>>>━━━━━━━━━━━━━ 2
            │                               │
            │                               │
            ↑                               ↓
            ↑               5               ↓
            ↑          ↗                   ↓
            │      ↗                       │
            │  ↗                           │
            4 ━━━━━━━━━━━━━<<<━━━━━━━━━━━━━ 5


        You might hear "beep" sound for each stage.
        After collecting data, pyPupil clusters the datum and finds the average position of each gaze.
        Then it applies Affine transform with fixed point.
        Finally, calibration returns the Affine Transform Matrices. (1 or 2 matrices)


        '''
        sub_socket, time0 = self._start_connection()

        # Coordinate transformation # second
        max_num_points = self.duration_calibrate * Pupil.frequency * self.num_cal_points * len(eye_to_clb)
        data_calibration = np.empty([max_num_points, 4])

        # Initialization
        t = 0
        position = 0
        index = 0
        global_idx = 0
        X = [np.empty(shape = [0, 2]), np.empty(shape = [0, 2])] # data for left and right eye
        indices_eye_position_change = [[],[]]

        # Data collecting
        while t < self.duration_calibrate * self.num_cal_points :
            topic, msg = sub_socket.recv_multipart()
            pupil_position = loads(msg)
            x, y = pupil_position[b'norm_pos']
            t = pupil_position[b'timestamp'] - time0
            conf = pupil_position[b'confidence']
            left_eye = int(str(topic)[self.idx_left_eye])

            new_position = int( t / self.duration_calibrate )
            if new_position > position :
                self._beep() # change gaze position with beep sound

                position = new_position # update position
                if position == self.num_cal_points :
                    break

                for eye in eye_to_clb:
                    indices_eye_position_change[eye].append(len(X[eye]))
                index = 0 # initialize index

            # START - Dummy processing...
            t_from_new_position = t - self.duration_calibrate * new_position

            if USE_DUMMY_PERIOD and t_from_new_position < self.dummy_period:
                # TODO : discard data
                print("%s at %.3fs | pupil position : (%.3f,%.3f)" % (topic, t, x, y) + " ** Will be treated as dummy **")
                pass

            else:
            # END - Dummy processing

                # START - Confidence test
                if conf < Pupil.conf_th_calib:
                    print("Deleted because of low confidence", conf)
                    continue


                # END - Confidence test

                data_calibration[global_idx, :] = [t, x, y, left_eye] # for matlab
                X[left_eye] = np.append(X[left_eye], [[x, y]], axis = 0) # for later data processing

                index = index + 1
                global_idx = global_idx + 1
                print("%s at %.3fs | pupil position : (%.3f,%.3f), conf:%.3f" % (topic, t, x, y, conf))

        sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))


        from_points = [[], []]

        # postprocessing eye by eye
        for eye in eye_to_clb:
            # (1) get points via k-mean clustering
            cluster = KMeans(n_clusters = self.num_cal_points, random_state = 0).fit(X[eye])
            # TODO : plot matplot will be convenient

            # (2) index(label) lookup table
            lut = self._idx_lut(cluster.labels_, indices_eye_position_change[eye])
            print("lookup table : ", lut)

            # (3) get centers from cluster and reorder
            from_points[eye] = [ cluster.cluster_centers_[i] for i in lut ]
            print("clustered point : ", from_points[eye])

            # (4) affine fitting (calibration)
            self.Affine_Transforms[eye] = Affine_Fit(from_points[eye], self.to_points)
            if self.Affine_Transforms[eye] == False:
                print("Not clustered well, Try again")
                return

            print("Affine Transform is ")
            print(self.Affine_Transforms[eye].To_Str())


        # save data into .mat format
        current_time = str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
        file_name = 'eye_track_before_calib_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data_calibration)
        self._save_file('eye_track_before_calib_data_latest.mat', data_calibration)

        # TEMP : save after transform data for visualization
        data_refine = np.empty([max_num_points * self.num_cal_points, 4])

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


    def get_calibration_frame(self):
        return self.to_points


    def get_duration(self, type):
        if type == 'calibration':
            return self.duration_calibrate
        elif type == 'record':
            return self.duration_record
        else:
            print("Unbeseeming type")
            return 0.0


    def record(self, duration = 20.0, synchronize = False):
        '''
        :param float duration: duration for Recording
        :param bool synchonize: whether to synchronize data from both eye (average)

        :returns: The 2D Array of recorded data

        Procedure
        -----------------
        1. receive Pupil data from device
        2. transfrom the pupil position to gaze new_position with Affine transform matrix with precaculated
        3. (If syncronize option is selected)
           Synchronize both eyes' data with average.

        '''
        # check whether calibrated and make connection
        if any(self.Affine_Transforms) is False :
            print("You should calibrate before record.")
            return

        sub_socket, time0 = self._start_connection()
        """
        self.sub_socket.connect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port) )

        # Find initial point of time.
        topic, msg = self.sub_socket.recv_multipart()
        pupil_position = loads(msg)
        time0 = pupil_position[b'timestamp']
        """

        # variable initialization
        max_num_points = duration * Pupil.frequency * 2
        data = np.empty([max_num_points, 7])
        t = 0
        index = 0

        # Recording starts with Beep sound
        self._beep()
        # Data acquisition with synchonization (left eye and right eye)
        qs = [Queue()]
        if synchronize:
            self.data = np.empty([max_num_points, 3])
            qs.append(Queue())
            self._synchronize(qs, 0, time.time())

        # Data acquisition from Pupil-labs Eye tracker
        while t < duration:
            topic, msg = sub_socket.recv_multipart()
            pupil_position = loads(msg)
            conf = pupil_position[b'confidence']
            coord = pupil_position[b'norm_pos']

            if conf < Pupil.conf_th_record :
                print("Deleted because of low confidence", conf)
                continue

            left_eye = int(str(topic)[self.idx_left_eye]) # 1 : left, 0 : right

            # get time and real coordinate with Affine Transform
            x, y = self.Affine_Transforms[left_eye].Transform(coord)
            t = pupil_position[b'timestamp'] - time0

            data[index, :] = [t, x, y, left_eye, coord[0], coord[1], conf]
            index = index + 1

            # Put queue due to synchronization
            if synchronize:
                qs[left_eye].put([t, x, y])
            else:
                print("%s at %.3fs | gaze position : (%.3f,%.3f), conf:%.3f" % (topic, t, x, y, conf))

        # Recoring finishes with Beep sound
        self._beep()
        sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))

        data_dict = {'timestamp' : np.zeros(1), \
                     'x' : np.zeros(1), \
                     'y' : np.zeros(1)}

        if synchronize:
            # Send synchronization end signal
            qs.append(None)
            for q in qs:
                if q is not None:
                    q.join()
            print("Thread finished..")

            data_dict['timestamp'] = self.data[:, 0]
            data_dict['x'] = self.data[:, 1]
            data_dict['y'] = self.data[:, 2]

        # Get raw data : ** Will be deprecated **
        data_raw = {}
        data_raw['timestamp'] = data[:, 0]
        data_raw['x_transformed'] = data[:, 1]
        data_raw['y_transformed'] = data[:, 2]
        data_raw['left_eye'] = data[:, 3]
        data_raw['x_raw'] = data[:, 4]
        data_raw['y_raw'] = data[:, 5]
        data_raw['conf'] = data[:, 6]

        data_dict['raw'] = data_raw

        return data_dict


    def set_calibration_frame(self, new_points):
        '''
        :param list new_points: list of points to calibrate
        '''
        # TODO : check whether new_points's type is numpy 2d array

        self.to_points = new_points
        self.num_cal_points = self.to_points.shape[0]


    def set_duration(self, type, duration):
        """
        :param string type: calibration or record
        :param float duration: duration for calibration or record
        """
        if type == 'calibration':
            self.duration_calibrate = duration
        elif type == 'record':
            self.duration_record = duration


    def save_data_dict(self, file_name, data_dict, object_name = 'data'):
        '''
        :param string file_name: name of file
        :param dict data_dict: dictionary that is from pypupil.
        :param string object_name: name of data in MATLAB.

        save data in .mat format with file name,
        You can put data as dictionary type, which is given py pypupil.
        This method will automatically convert the data into 2d array with given column order.
        '''
        if len(data_dict) < 4:
            data = np.column_stack((data_dict['timestamp'], \
                                    data_dict['x'], \
                                    data_dict['y']))
        else:
            pass

        file_dir = 'data/'
        file_name = file_dir + file_name
        scipy.io.savemat(file_name, mdict = { object_name : data })


    def throw(self, duration = 1000):
        '''
        :param float duration : duration

        Get processed data in real time
        '''
        # check whether calibrated and make connection
        if any(self.Affine_Transforms) is False :
            print("You should calibrate before record.")
            return

        sub_socket, time0 = self._start_connection()

        # Variable initialization
        max_num_points = duration * Pupil.frequency * 2
        t = 0
        self._beep()

        # Get raw data : ** Will be deprecated **
        data = np.empty([max_num_points, 7])

        # Data acquisition with synchonization (left eye and right eye)
        qs = [Queue(), Queue()]
        self._synchronize(qs, 0, time.time())

        # Data acquisition from Pupil-labs Eye tracker
        while t < duration:
            topic, msg = sub_socket.recv_multipart()
            pupil_position = loads(msg)
            conf = pupil_position[b'confidence']
            coord = pupil_position[b'norm_pos']

            # discard low-confidence data
            if conf < Pupil.conf_th_record :
                print("Deleted because of low confidence", conf)
                continue

            left_eye = int(str(topic)[self.idx_left_eye]) # 1 : left, 0 : right

            # get time and real coordinate with Affine Transform
            t = pupil_position[b'timestamp'] - time0
            x, y = self.Affine_Transforms[left_eye].Transform(coord)

            # Put queue due to synchronization
            qs[left_eye].put([t, x, y])
            print("%s at %.3fs | gaze position : (%.3f,%.3f), conf:%.3f" % (topic, t, x, y, conf))

        # Recoring finishes with Beep sound
        self._beep()
        sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))

        # Send synchronization END signal
        qs.append(None)
        for q in qs:
            if q is not None:
                q.join()

        print("Thread finished..")
        print("Throwing data finished..")



    def _synchronize(self, qs, index_sync, t0 = 0.0, prev_point = None):
        '''
        start synchronization of asynchronous eye data from both eyes
        It produces synchonized data every (1/120Hz) second using queue.

        Arguments :
            qs : list of queues of points received by pupil
                qs[0] : queue of datum from right eye
                qs[1] : queue of datum from left eye

            index_sync : it increases as this function called

            t0 : reference time from syncronization start

            prev_point : if one of both eye data does not arrive, We use the data which came just before.
        '''
        if None in qs:
            print("Thread finishing..")
            return

        #assert len(qs) == 2
        t_sync = index_sync * Pupil.period # time to synchronize
        t0_process = time.time()

        # Variable initialization
        qsizes = [ q.qsize() for q in qs ]
        qsize_min = min(qsizes)
        qsize_diff = max(qsizes) - qsize_min
        idx_qsize_max = np.argmax(qsizes)
        t_diff_max = 0.0

        t, x, y = 0.0, 0.0, 0.0
        n = 2 * qsize_min

        # Pop from queue and get average
        if qsize_min == 0 :
            if prev_point is not None:
                t, x, y = prev_point
        else:
            for i in range(qsize_min):
                ts = []
                for q in qs:
                    top = q.get_nowait()
                    ts.append(top[0])
                    t, x, y = t + top[0], x + top[1], y + top[2]

                if abs(ts[1] - ts[0]) > t_diff_max:
                    t_diff_max = abs(ts[1] - ts[0])

            t, x, y = t / n, x / n, y / n
            prev_point = t, x, y

            # Flush if t_diff_max > 8.3 ms
            if t_diff_max > Pupil.period :
                q = qs[idx_qsize_max]
                for i in range(qsize_diff):
                    q.get_nowait()

        t_real = time.time() - t0
        t_process = time.time() - t0_process
        t_delay = t_sync - t_real

        # Wait(synchronize) and restart thread
        thread_sync = threading.Timer(t_delay, self._synchronize, (qs, index_sync + 1, t0, prev_point))
        thread_sync.daemon = True
        thread_sync.start()

        self.data[index_sync, :] = [t_real, x, y] # for matlab
        print("t_sync : %.3f, t_real : %.3f | gaze position : (%.3f,%.3f)" % (t_sync, t_real, x, y) )

        t = t_real
        ############################# THROW DATA HERE #############################\
        # You can handle processed coordinate with timestamp here in real time
        # ex.) your_class.your_method(t,x,y)

        ###########################################################################


    def _plot_graph(self, data = None):
        data = np.empty(shape = [10, 2])

        for i in range(10):
            x, y = (i, 2*i)
            data[i, :] = [x, y]

        # Change the line plot below to a scatter plot
        print(data)
        plt.scatter(data[:, 0], data[:, 1])
        # Show plot
        plt.show()


    def _save_file(self, file_name, data, object_name = 'data'):
        '''
        save data in .mat format with file_name
        You can change the directory which the file will be saved.
        '''
        file_dir = 'data/'
        file_name = file_dir + file_name
        scipy.io.savemat(file_name, mdict = { object_name : data })


    def _idx_lut(self, labels, index_change):
        '''
        :param list labels: list of labels
        :param list index_change : list of indices that change gaze position

        :returns: Lookup Table of idx of n_clusters

        This is necessary because K-mean clustering does not guarantee
        the order of centers which I intended.

        This method get mode of labels in subarray then make lookuptable.

        get mode of labels in subarray
        and make lookup Table
        '''

        num_points = len(index_change) + 1
        spl = np.split(labels, index_change)
        LUT = np.empty(num_points, dtype = int)

        # Find mode value
        for i in range(num_points):
            points = np.array([spl[i]]).T
            mode_index = stats.mode(points)[0][0]
            LUT[i] = mode_index[0]

        return LUT


    def _beep(self):
        print('\a')


    def _start_connection(self):

        self._beep()
        self.sub_socket.connect(b"tcp://%s:%s" %(Pupil.addr_localhost.encode('utf-8'), self.sub_port))
        topic, msg = self.sub_socket.recv_multipart()

        # get data
        pupil_position = loads(msg)
        return self.sub_socket, pupil_position[b'timestamp']










    """
    Will be deprecated
    """
    def _record(self, synchronize = False):
        '''
        Make matlab file with recorded data, raw data
        Return nothing

        Procedure :
        1. receive Pupil data from device
        2. Transfrom the pupil position to gaze new_position
               With Affine transform matrix with precaculated
        3(If syncronize option is selected).
           Synchronize both eyes' data with average.
        '''

        # check whether calibrated and make connection
        if any(self.Affine_Transforms) is False :
            print("You should calibrate before record.")
            return
        self.sub_socket.connect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port) )

        # Find initial point of time.
        topic, msg = self.sub_socket.recv_multipart()
        pupil_position = loads(msg)
        time0 = pupil_position[b'timestamp']

        # Make null arrays to fill.
        max_num_points = self.duration_record * Pupil.frequency * 2
        data = np.empty([max_num_points, 6]) # Will be deprecated

        # variable initialization
        t = 0
        index = 0

        # Recording starts with Beep sound
        self._beep()

        # Data acquisition with synchonization (left eye and right eye)
        qs = [Queue()]
        if synchronize:
            self.data = np.empty([max_num_points, 3])
            qs.append(Queue())
            self._synchronize(qs, 0, time.time())

        # Data acquisition from Pupil-labs Eye tracker
        while t < self.duration_record:
            topic, msg = self.sub_socket.recv_multipart()

            pupil_position = loads(msg)
            coord = pupil_position[b'norm_pos']
            conf = pupil_position[b'confidence']


            if conf < Pupil.conf_th_record :
                print("Deleted because of low confidence", conf)
                continue


            left_eye = int(str(topic)[self.idx_left_eye]) # 1 : left, 0 : right

            # get real coordinate with Affine Transform
            x, y = self.Affine_Transforms[left_eye].Transform(coord)
            # get time
            t = pupil_position[b'timestamp'] - time0

            data[index, :] = [t, x, y, left_eye, coord[0], coord[1]]
            index = index + 1

            # Put queue due to synchronization
            if synchronize:
                qs[left_eye].put([t, x, y])
            else:
                print("%s at %.3fs | gaze position : (%.3f,%.3f), conf:%.3f" % (topic, t, x, y, conf))

        # Recoring finishes with Beep sound
        self._beep()
        self.sub_socket.disconnect(b"tcp://%s:%s" % (Pupil.addr_localhost.encode('utf-8'), self.sub_port))


        current_time = str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
        if synchronize:
            # send synchronization end signal
            qs.append(None)
            print("Thread finished..")
            file_name = 'eye_track_gaze_processed_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
            self._save_file(file_name, self.data)
            self._save_file('eye_track_gaze_processed_data_latest.mat', self.data)
            print("processed data saving...")

        # Convert and save MATLAB file
        file_name = 'eye_track_gaze_raw_data_' + current_time + '.mat' # file name ex : eye_track_data_180101_120847.mat
        self._save_file(file_name, data)
        self._save_file('eye_track_gaze_raw_data_latest.mat', data)
        print("raw data saving...")


"""

Open-source lib usage
---------------------
scipy, sklearn, numpy, zmq, msgpack, affine_transformer

"""
