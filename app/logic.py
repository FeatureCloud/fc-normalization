import os
import pickle
import threading
import time
import yaml

import pandas as pd
import numpy as np

from distutils import dir_util


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for master, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.master = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.train = None
        self.sep = None
        self.label_column = None
        self.data = None
        self.data_without_label = None
        self.mode = None

        self.filename = None
        self.values = None

        self.global_mean = None
        self.global_stddev = None
        self.global_min = None
        self.global_max = None
        self.global_maxabs = None

    def handle_setup(self, client_id, master, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.master = master
        self.clients = clients
        print(f'Received setup: {self.id} {self.master} {self.clients}')

        self.read_config()

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        print(f'Received data: {data}')
        # This method is called when new data arrives
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print(f'Submit data: {self.data_outgoing}')
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_normalization']
            self.train = config['files']['input']
            self.sep = config['files']['sep']
            self.label_column = config['files']['label_column']
            self.mode = config['files'].get('mode', 'variance')

    def app_flow(self):
        # This method contains a state machine for the slave and master instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_compute_local = 2
        state_gather = 3
        state_wait = 4
        state_result_ready = 5
        state_finishing = 6

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:

            if state == state_initializing:
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input

            # COMMON PART

            if state == state_read_input:
                print('Reading input...')
                self.data = pd.read_csv(os.path.join(f'/mnt/input/', self.train), sep=self.sep)
                self.data_without_label = self.data.drop(self.label_column, axis=1)
                print('Read input.')
                state = state_compute_local

            if state == state_compute_local:
                print('Calculate local values...')
                self.values = self.data_without_label.to_numpy()
                local_matrix = None
                if self.mode == 'variance':
                    local_matrix = np.zeros((self.values.shape[1], 3))
                    local_matrix[:, 0] = self.values.shape[0]  # Sample sizes
                    local_matrix[:, 1] = np.sum(np.square(self.values), axis=0)
                    local_matrix[:, 2] = np.sum(self.values, axis=0)
                elif self.mode == 'minmax':
                    local_matrix = np.zeros((self.values.shape[1], 2))
                    local_matrix[:, 0] = np.min(self.values, axis=0)
                    local_matrix[:, 1] = np.max(self.values, axis=0)
                elif self.mode == 'maxabs':
                    local_matrix = np.zeros((self.values.shape[1], 1))
                    local_matrix[:, 0] = np.max(np.abs(self.values), axis=0)
                print(f'Calculated local values: {local_matrix}')

                if self.master:
                    self.data_incoming.append(local_matrix.dumps())
                    state = state_gather
                else:
                    self.data_outgoing = local_matrix.dumps()
                    self.status_available = True
                    state = state_wait

            if state == state_result_ready:
                result = None
                if self.mode == 'variance':
                    result = (self.values - self.global_mean) / self.global_stddev
                elif self.mode == 'minmax':
                    result = (self.values - self.global_min) / (self.global_max - self.global_min)
                elif self.mode == 'maxabs':
                    result = self.values / self.global_maxabs

                result_df = pd.DataFrame(data=result, columns=self.data_without_label.columns)
                result_df[self.label_column] = self.data.loc[:, self.label_column]
                result_df.to_csv(os.path.join('/mnt/output/', self.train), index=False, sep=self.sep)

                if self.master:
                    state = state_finishing
                else:
                    break

            # GLOBAL PART

            if state == state_gather:
                if len(self.data_incoming) == len(self.clients):
                    global_matrix = np.zeros((self.values.shape[1], 3))

                    for local_matrix_bytes in self.data_incoming:
                        local_matrix = np.loads(local_matrix_bytes)
                        if self.mode == 'variance':
                            global_matrix[:, 0] += local_matrix[:, 0]
                            global_matrix[:, 1] += local_matrix[:, 1]
                            global_matrix[:, 2] += local_matrix[:, 2]
                        elif self.mode == 'minmax':
                            global_matrix[:, 0] = \
                                np.min(np.stack([global_matrix[:, 0], local_matrix[:, 0]], axis=1), axis=1)
                            global_matrix[:, 1] = \
                                np.max(np.stack([global_matrix[:, 1], local_matrix[:, 1]], axis=1), axis=1)
                        elif self.mode == 'maxabs':
                            global_matrix[:, 0] = \
                                np.max(np.stack([global_matrix[:, 0], local_matrix[:, 0]], axis=1), axis=1)

                    if self.mode == 'variance':
                        global_mean_square = global_matrix[:, 1] / global_matrix[:, 0]
                        global_mean = global_matrix[:, 2] / global_matrix[:, 0]
                        global_stddev = np.sqrt(global_mean_square - np.square(global_mean))
                        self.data_outgoing = pickle.dumps({
                            'stddev': global_stddev,
                            'mean': global_mean,
                        })
                        print(f'Mean: {global_mean}')
                        print(f'Variance: {global_stddev}')
                        self.global_mean = global_mean
                        self.global_stddev = global_stddev
                    elif self.mode == 'minmax':
                        global_min = global_matrix[:, 0]
                        global_max = global_matrix[:, 1]
                        self.data_outgoing = pickle.dumps({
                            'min': global_min,
                            'max': global_max,
                        })
                        print(f'Min: {global_min}')
                        print(f'Max: {global_max}')
                        self.global_min = global_min
                        self.global_max = global_max
                    elif self.mode == 'maxabs':
                        global_maxabs = global_matrix[:, 0]
                        self.data_outgoing = pickle.dumps({
                            'maxabs': global_maxabs,
                        })
                        print(f'Maxabs: {global_maxabs}')
                        self.global_maxabs = global_maxabs

                    self.status_available = True
                    state = state_result_ready
                else:
                    print(f'Have {len(self.data_incoming)} of {len(self.clients)} so far, waiting...')

            if state == state_finishing:
                time.sleep(10)
                self.status_finished = True
                break

            # LOCAL PART

            if state == state_wait:
                if len(self.data_incoming) > 0:
                    pkg = pickle.loads(self.data_incoming[0])
                    if self.mode == 'variance':
                        self.global_stddev = pkg['stddev']
                        self.global_mean = pkg['mean']
                        print(f'Stddev: {self.global_stddev}')
                        print(f'Mean: {self.global_mean}')
                    elif self.mode == 'minmax':
                        self.global_min = pkg['min']
                        self.global_max = pkg['max']
                        print(f'Min: {self.global_min}')
                        print(f'Max: {self.global_max}')
                    elif self.mode == 'maxabs':
                        self.global_maxabs = pkg['maxabs']
                        print(f'Maxabs: {self.global_maxabs}')

                    state = state_result_ready

            time.sleep(1)


logic = AppLogic()
