import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import threading
import time
import yaml
from distutils import dir_util

APP_NAME = 'fc_normalization'


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
        self.input_train = None
        self.input_test = None
        self.output_train = None
        self.output_test = None
        self.split_mode = None
        self.split_dir = None

        self.sep = None
        self.label = None

        self.mode = None
        self.normalize_test = None

        self.data = []
        self.data_without_label = []

        self.filename = None

        self.values = []
        self.global_mean = []
        self.global_stddev = []
        self.global_min = []
        self.global_max = []
        self.global_maxabs = []

        self.lock = threading.Lock()

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
        # This method is called when new data arrives
        self.lock.acquire()
        self.data_incoming.append(data.read())
        self.lock.release()

    def handle_outgoing(self):
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[APP_NAME]

            self.input_train = config['input'].get('train')
            self.input_test = config['input'].get('test')

            self.output_train = config['output'].get('train')
            self.output_test = config['output'].get('test')

            self.split_mode = config['split'].get('mode')
            self.split_dir = config['split'].get('dir')

            self.sep = config['format']['sep']
            self.label = config['format']['label']

            self.mode = config.get('normalization', 'variance')
            self.normalize_test = config.get('normalize_test', False)

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

                def read_input(ins, path):
                    d = pd.read_csv(path, sep=ins.sep)
                    ins.data.append(d)
                    ins.data_without_label.append(d.drop(self.label, axis=1))

                print('Reading input...')
                base_dir = os.path.normpath(os.path.join(f'/mnt/input/', self.split_dir))
                if self.split_mode == 'directory':
                    for split_name in os.listdir(base_dir):
                        read_input(self, os.path.join(base_dir, split_name, self.input_train))
                        if self.input_test is not None:
                            read_input(self, os.path.join(base_dir, split_name, self.input_test))
                elif self.split_mode == 'file':
                    read_input(self, os.path.join(base_dir, self.input_train))
                    if self.input_test is not None:
                        read_input(self, os.path.join(base_dir, self.input_test))

                print('Read input.')
                state = state_compute_local

            if state == state_compute_local:
                print('Calculate local values...')
                local_matrices = []
                for dwl in self.data_without_label:
                    values = dwl.to_numpy()
                    self.values.append(values)
                    local_matrix = None
                    if self.mode == 'variance':
                        local_matrix = np.zeros((values.shape[1], 3))
                        local_matrix[:, 0] = values.shape[0]  # Sample sizes
                        local_matrix[:, 1] = np.sum(np.square(values), axis=0)
                        local_matrix[:, 2] = np.sum(values, axis=0)
                    elif self.mode == 'minmax':
                        local_matrix = np.zeros((values.shape[1], 2))
                        local_matrix[:, 0] = np.min(values, axis=0)
                        local_matrix[:, 1] = np.max(values, axis=0)
                    elif self.mode == 'maxabs':
                        local_matrix = np.zeros((values.shape[1], 1))
                        local_matrix[:, 0] = np.max(np.abs(values), axis=0)
                    local_matrices.append(local_matrix)

                print(f'Calculated local values.')

                if self.master:
                    self.lock.acquire()
                    self.data_incoming.append(pickle.dumps(local_matrices))
                    self.lock.release()
                    state = state_gather
                else:
                    self.data_outgoing = pickle.dumps(local_matrices)
                    self.status_available = True
                    state = state_wait

            if state == state_result_ready:

                results = []
                for i in range(len(self.values)):
                    norm_i = i  # Index for choosing normalization values
                    if i % 2 == 1 and self.input_test and not self.normalize_test:
                        norm_i = i - 1

                    if self.mode == 'variance':
                        a = (self.values[i] - self.global_mean[norm_i])
                        b = self.global_stddev[norm_i]
                        normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                    elif self.mode == 'minmax':
                        a = (self.values[i] - self.global_min[norm_i])
                        b = (self.global_max[norm_i] - self.global_min[norm_i])
                        normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                    elif self.mode == 'maxabs':
                        a = self.values[i]
                        b = self.global_maxabs[norm_i]
                        normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                    normalized[normalized == np.inf] = 0
                    normalized[normalized == -np.inf] = 0
                    normalized[normalized == np.nan] = 0
                    results.append(normalized)

                def write_output(ins, result, path):
                    result_df = pd.DataFrame(data=result[0], columns=ins.data_without_label[0].columns)
                    result_df[ins.label] = ins.data[0].loc[:, ins.label]
                    result_df.to_csv(path, index=False, sep=ins.sep)

                    ins.data = ins.data[1:]
                    ins.data_without_label = ins.data_without_label[1:]
                    result = result[1:]

                    return result

                print('Writing output...')
                base_dir_in = os.path.normpath(os.path.join(f'/mnt/input/', self.split_dir))
                base_dir_out = os.path.normpath(os.path.join(f'/mnt/output/', self.split_dir))
                if self.split_mode == 'directory':
                    for split_name in os.listdir(os.path.join(base_dir_in)):
                        out_path = os.path.join(base_dir_out, split_name)
                        pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
                        results = write_output(self, results, os.path.join(out_path, self.output_train))
                        if self.input_test is not None:
                            results = write_output(self, results, os.path.join(out_path, self.output_test))
                elif self.split_mode == 'file':
                    out_path = base_dir_out
                    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
                    results = write_output(self, results, os.path.join(out_path, self.output_train))
                    if self.input_test is not None:
                        results = write_output(self, results, os.path.join(out_path, self.output_test))

                if self.master:
                    self.data_incoming = ['DONE']
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            # GLOBAL PART

            if state == state_gather:
                if len(self.data_incoming) == len(self.clients):
                    print(f'Have everything, continuing...')

                    client_data = []
                    for local_matrix_bytes in self.data_incoming:
                        client_data.append(pickle.loads(local_matrix_bytes))

                    data_outgoing = []

                    for i in range(len(self.values)):
                        global_matrix = np.zeros((self.values[i].shape[1], 3))

                        for local_matrix in client_data:
                            if self.mode == 'variance':
                                global_matrix[:, 0] += local_matrix[i][:, 0]
                                global_matrix[:, 1] += local_matrix[i][:, 1]
                                global_matrix[:, 2] += local_matrix[i][:, 2]
                            elif self.mode == 'minmax':
                                global_matrix[:, 0] = \
                                    np.min(np.stack([global_matrix[:, 0], local_matrix[i][:, 0]], axis=1), axis=1)
                                global_matrix[:, 1] = \
                                    np.max(np.stack([global_matrix[:, 1], local_matrix[i][:, 1]], axis=1), axis=1)
                            elif self.mode == 'maxabs':
                                global_matrix[:, 0] = \
                                    np.max(np.stack([global_matrix[:, 0], local_matrix[i][:, 0]], axis=1), axis=1)

                        if self.mode == 'variance':
                            global_mean_square = global_matrix[:, 1] / global_matrix[:, 0]
                            global_mean = global_matrix[:, 2] / global_matrix[:, 0]
                            global_stddev = np.sqrt(global_mean_square - np.square(global_mean))
                            data_outgoing.append({
                                'stddev': global_stddev,
                                'mean': global_mean,
                            })
                            self.global_mean.append(global_mean)
                            self.global_stddev.append(global_stddev)
                        elif self.mode == 'minmax':
                            global_min = global_matrix[:, 0]
                            global_max = global_matrix[:, 1]
                            data_outgoing.append({
                                'min': global_min,
                                'max': global_max,
                            })
                            self.global_min.append(global_min)
                            self.global_max.append(global_max)
                        elif self.mode == 'maxabs':
                            global_maxabs = global_matrix[:, 0]
                            data_outgoing.append({
                                'maxabs': global_maxabs,
                            })
                            self.global_maxabs.append(global_maxabs)

                    self.data_outgoing = pickle.dumps(data_outgoing)
                    self.status_available = True
                    state = state_result_ready
                else:
                    print(f'Have {len(self.data_incoming)} of {len(self.clients)} so far, waiting...')

            if state == state_finishing:
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            # LOCAL PART

            if state == state_wait:
                if len(self.data_incoming) > 0:
                    pkg = pickle.loads(self.data_incoming[0])
                    for i in range(len(self.values)):
                        if self.mode == 'variance':
                            self.global_stddev.append(pkg[i]['stddev'])
                            self.global_mean.append(pkg[i]['mean'])
                        elif self.mode == 'minmax':
                            self.global_min.append(pkg[i]['min'])
                            self.global_max.append(pkg[i]['max'])
                        elif self.mode == 'maxabs':
                            self.global_maxabs.append(pkg[i]['maxabs'])

                    state = state_result_ready

            time.sleep(1)


logic = AppLogic()
