import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import threading
import time
import yaml
from distutils import dir_util

from FeatureCloud.app.engine.app import AppState, app_state, Role

APP_NAME = 'fc_normalization'

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log(f"[CLIENT] Coordinator {self.is_coordinator}")
        
        return 'read input'


@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """

    def register(self):
        self.register_transition('compute local', Role.BOTH)
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Read input and config")
        self.read_config()
            
        def read_input(path):
            d = pd.read_csv(path, sep=self.load('sep'))
            return d, d.drop(self.load('label'), axis=1)
  
        base_dir = os.path.normpath(os.path.join(f'/mnt/input/', self.load('split_dir')))
            
        data = []
        data_without_label = []
            
        if self.load('split_mode') == 'directory':
            for split_name in os.listdir(base_dir):
                d, d_without_label = read_input(os.path.join(base_dir, split_name, self.load('input_train')))
                data.append(d)
                data_without_label.append(d_without_label)
                if self.load('input_test') is not None:
                    d, d_without_label = read_input(os.path.join(base_dir, split_name, self.load('input_test')))
                    data.append(d)
                    data_without_label.append(d_without_label)
        elif self.load('split_mode') == 'file':
            d, d_without_label = read_input(os.path.join(base_dir, self.load('input_train')))
            data.append(d)
            data_without_label.append(d_without_label)
            if self.load('input_test') is not None:
                d, d_without_label = read_input(os.path.join(base_dir, self.load('input_test')))
                data.append(d)
                data_without_label.append(d_without_label)
                
        self.store('data', data)
        self.store('data_without_label', data_without_label)
        return 'compute local'
 
 
    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[APP_NAME]

            self.store('input_train', config['input'].get('train'))
            self.store('input_test', config['input'].get('test'))

            self.store('output_train', config['output'].get('train'))
            self.store('output_test', config['output'].get('test'))

            self.store('split_mode', config['split'].get('mode'))
            self.store('split_dir', config['split'].get('dir'))

            self.store('sep', config['format']['sep'])
            self.store('label', config['format']['label'])

            self.store('mode', config.get('normalization', 'variance'))
            self.store('normalize_test', config.get('normalize_test', False))

            
# COMMON PART
@app_state('compute local', Role.BOTH)
class ComputeLocalState(AppState):
    """
    Perform local computation and send the computation data to the coordinator.
    """

    def register(self):
        self.register_transition('gather', Role.COORDINATOR)
        self.register_transition('wait', Role.PARTICIPANT)
        
    def run(self) -> str or None:

        self.log('Calculate local values...')
        local_matrices = []
        values = []
        for dwl in self.load('data_without_label'):
            local_values = dwl.to_numpy()
            values.append(local_values)
            local_matrix = None
            if self.load('mode') == 'variance':
                local_matrix = np.zeros((local_values.shape[1], 3))
                local_matrix[:, 0] = local_values.shape[0]  # Sample sizes
                local_matrix[:, 1] = np.sum(np.square(local_values), axis=0)
                local_matrix[:, 2] = np.sum(local_values, axis=0)
            elif self.load('mode') == 'minmax':
                local_matrix = np.zeros((local_values.shape[1], 2))
                local_matrix[:, 0] = np.min(local_values, axis=0)
                local_matrix[:, 1] = np.max(local_values, axis=0)
            elif self.load('mode') == 'maxabs':
                local_matrix = np.zeros((local_values.shape[1], 1))
                local_matrix[:, 0] = np.max(np.abs(local_values), axis=0)
            local_matrices.append(local_matrix)
        self.store('values', values)
        self.log(f'Calculated local values.')
        self.send_data_to_coordinator(pickle.dumps(local_matrices))

        if self.is_coordinator:
            return 'gather'
        else:
            return 'wait'


@app_state('result ready', Role.BOTH)
class ResultReadyState(AppState):
    """
    Writes the results of the global computation.
    """

    def register(self):
        self.register_transition('finishing', Role.COORDINATOR)
        self.register_transition('terminal', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        results = []
        for i in range(len(self.load('values'))):
            norm_i = i  # Index for choosing normalization values
            if i % 2 == 1 and self.load('input_test') and not self.load('normalize_test'):
                norm_i = i - 1

            if self.load('mode') == 'variance':
                a = (self.load('values')[i] - self.load('global_mean')[norm_i])
                b = self.load('global_stddev')[norm_i]
                normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            elif self.load('mode') == 'minmax':
                a = (self.load('values')[i] - self.load('global_min')[norm_i])
                b = (self.load('global_max')[norm_i] - self.load('global_min')[norm_i])
                normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            elif self.load('mode') == 'maxabs':
                a = self.load('values')[i]
                b = self.load('global_maxabs')[norm_i]
                normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        
            normalized[normalized == np.inf] = 0
            normalized[normalized == -np.inf] = 0
            normalized[normalized == np.nan] = 0
            results.append(normalized)

        def write_output(result, path):
            self.log("Write output")
            result_df = pd.DataFrame(data=result[0], columns=self.load('data_without_label')[0].columns)
            result_df[self.load('label')] = self.load('data')[0].loc[:, self.load('label')]
            result_df.to_csv(path, index=False, sep=self.load('sep'))
            data = self.load('data')
            data_without_label = self.load('data_without_label')
            self.store('data', data[1:])
            self.store('data_without_label', data_without_label[1:])
            
            return result[1:]

        self.log('Prepare output...')
        base_dir_in = os.path.normpath(os.path.join(f'/mnt/input/', self.load('split_dir')))
        base_dir_out = os.path.normpath(os.path.join(f'/mnt/output/', self.load('split_dir')))
        if self.load('split_mode') == 'directory':
            for split_name in os.listdir(os.path.join(base_dir_in)):
                self.log(split_name)
                out_path = os.path.join(base_dir_out, split_name)
                pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
                results = write_output(results, os.path.join(out_path, self.load('output_train')))
                if self.load('input_test') is not None:
                    results = write_output(results, os.path.join(out_path, self.load('output_test')))
        elif self.load('split_mode') == 'file':
            out_path = base_dir_out
            pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
            results = write_output(results, os.path.join(out_path, self.load('output_train')))
            if self.load('input_test') is not None:
                results = write_output(results, os.path.join(out_path, self.load('output_test')))
        
        self.send_data_to_coordinator('DONE')
        
        if self.is_coordinator:
            return 'finishing'
        else:
            return 'terminal'


# GLOBAL PART
@app_state('gather', Role.COORDINATOR)
class GatherState(AppState):
    """
    The coordinator receives the local computation data from each client and aggregates it.
    The coordinator broadcasts the global computation data to the clients.
    """

    def register(self):
        self.register_transition('result ready', Role.COORDINATOR)
        
    def run(self) -> str or None:
        data = self.gather_data()
        self.log(f'Have everything, continuing...')

        client_data = []
        for local_matrix_bytes in data:
            client_data.append(pickle.loads(local_matrix_bytes))

        data_outgoing = []
        global_mean = []
        global_stddev = []
        global_min = []
        global_max = []
        global_maxabs = []
            
        for i in range(len(self.load('values'))):
            global_matrix = np.zeros((self.load('values')[i].shape[1], 3))

            for local_matrix in client_data:
                if self.load('mode') == 'variance':
                    global_matrix[:, 0] += local_matrix[i][:, 0]
                    global_matrix[:, 1] += local_matrix[i][:, 1]
                    global_matrix[:, 2] += local_matrix[i][:, 2]
                elif self.load('mode') == 'minmax':
                    global_matrix[:, 0] = \
                        np.min(np.stack([global_matrix[:, 0], local_matrix[i][:, 0]], axis=1), axis=1)
                    global_matrix[:, 1] = \
                        np.max(np.stack([global_matrix[:, 1], local_matrix[i][:, 1]], axis=1), axis=1)
                elif self.load('mode') == 'maxabs':
                    global_matrix[:, 0] = \
                        np.max(np.stack([global_matrix[:, 0], local_matrix[i][:, 0]], axis=1), axis=1)

            if self.load('mode') == 'variance':
                mean_square = global_matrix[:, 1] / global_matrix[:, 0]
                mean = global_matrix[:, 2] / global_matrix[:, 0]
                stddev = np.sqrt(mean_square - np.square(mean))
                data_outgoing.append({
                                'stddev': stddev,
                                'mean': mean,
                            })
                global_mean.append(mean)
                global_stddev.append(stddev)
            elif self.load('mode') == 'minmax':
                min = global_matrix[:, 0]
                max = global_matrix[:, 1]
                data_outgoing.append({
                                'min': min,
                                'max': max,
                            })
                global_min.append(min)
                global_max.append(max)
            elif self.load('mode') == 'maxabs':
                maxabs = global_matrix[:, 0]
                data_outgoing.append({
                                'maxabs': maxabs,
                            })
                global_maxabs.append(maxabs)

        self.store('global_mean', global_mean)
        self.store('global_stddev', global_stddev)
        self.store('global_min', global_min)
        self.store('global_max', global_max)
        self.store('global_maxabs', global_maxabs)

        self.broadcast_data(pickle.dumps(data_outgoing), send_to_self=False)
        return 'result ready'


@app_state('finishing', Role.COORDINATOR)
class FinishingState(AppState):

    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.gather_data()
        return 'terminal'

# LOCAL PART
@app_state('wait', Role.PARTICIPANT)
class WaitState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """

    def register(self):
        self.register_transition('result ready', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        data = self.await_data()
        pkg = pickle.loads(data)
        
        global_mean = []
        global_stddev = []
        global_min = []
        global_max = []
        global_maxabs = []
            
        for i in range(len(self.load('values'))):
            if self.load('mode') == 'variance':
                global_stddev.append(pkg[i]['stddev'])
                global_mean.append(pkg[i]['mean'])
            elif self.load('mode') == 'minmax':
                global_min.append(pkg[i]['min'])
                global_max.append(pkg[i]['max'])
            elif self.load('mode') == 'maxabs':
                global_maxabs.append(pkg[i]['maxabs'])
        
        self.store('global_mean', global_mean)
        self.store('global_stddev', global_stddev)
        self.store('global_min', global_min)
        self.store('global_max', global_max)
        self.store('global_maxabs', global_maxabs)
        
        return 'result ready'
