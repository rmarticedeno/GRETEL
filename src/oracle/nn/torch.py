import numpy as np
import torch

from src.core.oracle_base import Oracle
from src.core.torch_base import TorchBase
from src.dataset.dataset_base import Dataset
from src.n_dataset.utils.dataset_torch import TorchGeometricDataset
from src.oracle.nn.gcn import DownstreamGCN


class OracleTorch(TorchBase, Oracle):
                                
            
    def real_fit(self):
        super().real_fit()
        self.evaluate(self.dataset, fold_id=self.fold_id)
            
    @torch.no_grad()        
    def evaluate(self, dataset: Dataset, fold_id=0):            
        loader = dataset.get_torch_loader(fold_id=fold_id, batch_size=self.batch_size, usage='test')
        
        losses = []
        accuracy = []
        for batch in loader:
            node_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device)
            labels = batch.y.to(self.device)
            labels_n=torch.nn.functional.one_hot(labels, num_classes=self.dataset.num_classes).double()
            
            self.optimizer.zero_grad()  
            pred = self.fun(self.model(node_features, edge_index, edge_weights, batch.batch))
            
            loss = self.loss_fn(pred, labels_n)
            losses.append(loss.to('cpu').detach().numpy())
            
            pred_label = torch.argmax(pred,dim=1)
            accuracy += torch.eq(labels, pred_label).int().tolist()
        
        self.context.logger.info(f'Test accuracy ---> Test accuracy = {np.mean(accuracy):.4f}')


    def _real_predict(self, data_instance):
        return torch.argmax(self._real_predict_proba(data_instance))
    
    @torch.no_grad()
    def _real_predict_proba(self, data_inst):
        data_inst = TorchGeometricDataset.to_geometric(data_inst)

        node_features = data_inst.x.to(self.device)
        edge_index = data_inst.edge_index.to(self.device)
        edge_weights = data_inst.edge_attr.to(self.device)

        return self.model(node_features,edge_index,edge_weights, None).squeeze()
    
                     
    def check_configuration(self):#TODO: revise configuration
        super().check_configuration()
        local_config = self.local_config

        if 'model' not in local_config['parameters']:
            local_config['parameters']['model'] = {
                'class': "src.oracle.nn.gcn.DownstreamGCN",
                "parameters" : {}
            }

        # set defaults
        local_config['parameters']['model']['parameters']['node_features'] = self.dataset.num_node_features()
        local_config['parameters']['model']['parameters']['n_classes'] = self.dataset.num_classes