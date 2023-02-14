from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.explainer.ensemble.ensemble_factory import EnsembleFactory
from src.explainer.explainer_base import Explainer
from src.explainer.explainer_bidirectional_search import (
    DataDrivenBidirectionalSearchExplainer,
    ObliviousBidirectionalSearchExplainer)
from src.explainer.explainer_cfgnnexplainer import CFGNNExplainer
from src.explainer.explainer_dce_search import (DCESearchExplainer,
                                                DCESearchExplainerOracleless)
from src.explainer.explainer_maccs import MACCSExplainer
from src.explainer.explainer_countergan import CounteRGANExplainer


class ExplainerFactory:

    def __init__(self, explainer_store_path) -> None:
        self._explainer_id_counter = 0
        self._explainer_store_path = explainer_store_path
        self._ensemble_factory = EnsembleFactory(explainer_store_path, self)

    def get_explainer_by_name(self, explainer_dict, metric_factory : EvaluationMetricFactory) -> Explainer:
        explainer_name = explainer_dict['name']
        explainer_parameters = explainer_dict['parameters']

        # Check if the explainer is DCE Search
        if explainer_name == 'dce_search':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''DCE Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])

            # Returning the explainer
            return self.get_dce_search_explainer(dist_metric, explainer_dict)

        # Check if the explainer is DCE Search Oracleless
        elif explainer_name == 'dce_search_oracleless':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''DCE Search Oracleless requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_dce_search_explainer_oracleless(dist_metric, explainer_dict)

        elif explainer_name == 'bidirectional_oblivious_search':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''Bidirectional Oblivious Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_bidirectional_oblivious_search_explainer(dist_metric, explainer_dict)

        elif explainer_name == 'bidirectional_data-driven_search':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''Bidirectional Data-Driven Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_bidirectional_data_driven_search_explainer(dist_metric, explainer_dict)

        elif explainer_name == 'maccs':
            # Verifying the explainer parameters
            if not 'graph_distance' in explainer_parameters:
                raise ValueError('''Bidirectional Data-Driven Search requires a graph distance function''')

            # Getting the instance distance metric
            dist_metric = metric_factory.get_evaluation_metric_by_name(explainer_parameters['graph_distance'])
            
            # Returning the explainer
            return self.get_maccs_explainer(dist_metric, explainer_dict)

        
        elif explainer_name == 'cfgnnexplainer':
            # Returning the explainer
            return self.get_cfgnn_explainer(explainer_dict)

        
        elif explainer_name == 'ensemble':
            # Returning the ensemble explainer
            return self.get_ensemble(explainer_dict, metric_factory)
        
        elif explainer_name == 'countergan':
            # Verifying the explainer parameters
            if not 'n_nodes' in explainer_parameters:
                raise ValueError('''CounteRGAN requires the number of nodes''')
            if not 'batch_size_ratio' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a batch size ratio''')
            if not 'device' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a device''')
            if not 'training_iterations' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a number of training iterations''')
            if not 'real_label' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a real_label''')
            if not 'fake_label' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a fake_label''')
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a fold_id''')
            
            n_nodes = int(explainer_parameters['n_nodes'])
            batch_size_ratio =explainer_parameters['batch_size_ratio']
            device =explainer_parameters['device']
            training_iterations =explainer_parameters['training_iterations']
            real_label =explainer_parameters['real_label']
            fake_label =explainer_parameters['fake_label']
            fold_id =explainer_parameters['fold_id']
            
            return self.get_countergan_explainer(n_nodes, batch_size_ratio, device, training_iterations, real_label, fake_label, fold_id, explainer_dict)

        else:
            raise ValueError('''The provided explainer name does not match any explainer provided 
            by the factory''')


    def get_dce_search_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = DCESearchExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1  
        return result

    def get_dce_search_explainer_oracleless(self, instance_distance_function, config_dict=None) -> Explainer:
        result = DCESearchExplainerOracleless(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_bidirectional_oblivious_search_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = ObliviousBidirectionalSearchExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_bidirectional_data_driven_search_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = DataDrivenBidirectionalSearchExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_maccs_explainer(self, instance_distance_function, config_dict=None) -> Explainer:
        result = MACCSExplainer(self._explainer_id_counter, instance_distance_function, config_dict)
        self._explainer_id_counter += 1
        return result

    def get_cfgnn_explainer(self, config_dict=None) -> Explainer:
        result = CFGNNExplainer(self._explainer_id_counter, config_dict)
        self._explainer_id_counter += 1
        return result

    
    def get_ensemble(self, config_dic = None, metric_factory : EvaluationMetricFactory = None) -> Explainer:
        result = self._ensemble_factory.build_explainer(config_dic, metric_factory)
        self._explainer_id_counter += 1
        return result
    
    def get_countergan_explainer(self, n_nodes, batch_size_ratio, device, training_iterations, real_label, fake_label, fold_id, config_dict=None) -> Explainer:
        result = CounteRGANExplainer(self._explainer_id_counter, self._explainer_store_path, n_nodes, batch_size_ratio, device, training_iterations, real_label, fake_label, fold_id, config_dict)
        self._explainer_id_counter += 1
        return result
        
        
        