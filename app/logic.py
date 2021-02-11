import json
import threading
import time

import pandas as pd
import numpy as np


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
        self.data = None
        self.values = None
        self.global_variance = None

    def handle_setup(self, client_id, master, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.master = master
        self.clients = clients
        print(f'Received setup: {self.id} {self.master} {self.clients}')

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

    def app_flow(self):
        # This method contains a state machine for the slave and master instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_compute_variance = 2
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
                self.data = pd.read_csv('/mnt/input/input.csv')
                print('Read input.')
                state = state_compute_variance

            if state == state_compute_variance:
                print('Calculate local values...')
                self.values = self.data.to_numpy()
                local_matrix = np.zeros((self.values.shape[1], 3))
                local_matrix[:, 0] = self.values.shape[0]  # Sample sizes
                local_matrix[:, 1] = np.sum(np.square(self.values), axis=0)
                local_matrix[:, 2] = np.sum(self.values, axis=0)
                print(f'Calculated local values: {local_matrix}')

                if self.master:
                    self.data_incoming.append(local_matrix.dumps())
                    state = state_gather
                else:
                    self.data_outgoing = local_matrix.dumps()
                    self.status_available = True
                    state = state_wait

            if state == state_result_ready:
                result = self.values * self.global_variance
                result_df = pd.DataFrame(data=result, columns=self.data.columns)
                result_df.to_csv('/mnt/output/output.csv', index=False)

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
                        global_matrix[:, 0] += local_matrix[:, 0]
                        global_matrix[:, 1] += local_matrix[:, 1]
                        global_matrix[:, 2] += local_matrix[:, 2]

                    global_mean_square = global_matrix[:, 1] / global_matrix[:, 0]
                    global_square_mean = np.square(global_matrix[:, 2] / global_matrix[:, 0])
                    self.global_variance = global_mean_square - global_square_mean

                    print(f'Result: {self.global_variance}')

                    self.data_outgoing = self.global_variance.dumps()
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
                    self.global_variance = np.loads(self.data_incoming[0])
                    print(f'Result: {self.global_variance}')
                    state = state_result_ready

            time.sleep(1)


logic = AppLogic()
