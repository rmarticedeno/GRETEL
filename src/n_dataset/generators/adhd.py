from os import listdir
from os.path import isfile, join

import numpy as np

from src.n_dataset.instances.graph import GraphInstance
from src.n_dataset.generators.base import Generator


class ADHD(Generator):

    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        # Path to the instances of the "Typical development class"
        self.td_class_path = join(base_path, 'td')
        # Path to the instances of the "Autism Spectrum Disorder class"
        self.adhd_class_path = join(base_path, 'adhd_dataset')  
        self.generate_dataset()

    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.read_adjacency_matrices()
    
    def read_adjacency_matrices(self):
        """
        Reads the dataset from the adjacency matrices
        """
        
        paths = [self.td_class_path, self.adhd_class_path]

        instance_id = 0
        graph_label = 0
        # For each class folder
        for path in paths:
            for filename in listdir(path):
                # avoiding files not related to the dataset
                if 'DS_Store' not in filename:
                    # Reading the adjacency matrix
                    with open(path.join(path, filename), 'r') as f:
                        if filename[-3:]=='csv':
                            l = [[int(float(num)) for num in line.split(',')] for line in f] # if .csv 
                        else:
                            l = [[int(num) for num in line.split(' ')] for line in f] # if .txt

                        # Creating the instance
                        l_array = np.array(l)
                        inst = GraphInstance(instance_id, graph_label, l_array, dataset=self.dataset)
                        instance_id += 1    
                        #inst.name = filename.split('.')[0]
                        
                        # Adding the instance to the instances list
                        self.dataset.instances.append(inst)
            graph_label +=1