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
from src.explainer.explainer_clear import CLEARExplainer


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
            if not 'device' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a device''')
            if not 'n_labels' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a n_labels''')
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''CounteRGAN requires a fold_id''')
            
            n_nodes = int(explainer_parameters['n_nodes'])
            batch_size_ratio = explainer_parameters.get('batch_size_ratio', .1)
            device = explainer_parameters['device']
            training_iterations = explainer_parameters.get('training_iterations', 20000)
            n_generator_steps = explainer_parameters.get('n_generator_steps', 2)
            n_discriminator_steps = explainer_parameters.get('n_discriminator_steps', 3)
            n_labels = explainer_parameters['n_labels']
            fold_id = explainer_parameters['fold_id']
            ce_binarization_threshold = explainer_parameters.get('ce_binarization_threshold', None)
            
            return self.get_countergan_explainer(n_nodes, batch_size_ratio, device,
                                                 training_iterations, n_discriminator_steps,
                                                 n_generator_steps, n_labels, fold_id,
                                                 ce_binarization_threshold, explainer_dict)
            
        elif explainer_name == 'clear':
            # Verifying the explainer parameters
            if not 'n_nodes' in explainer_parameters:
                raise ValueError('''CLEAR requires the number of nodes''')
            if not 'n_labels' in explainer_parameters:
                raise ValueError('''CLEAR requires a n_labels''')
            if not 'fold_id' in explainer_parameters:
                raise ValueError('''CLEAR requires a fold_id''')
        
            batch_size_ratio = explainer_parameters.get('batch_size_ratio', .1)
            vae_type = explainer_parameters.get('vae_type', 'graphVAE')
            h_dim = explainer_parameters.get('h_dim', 16)
            z_dim = explainer_parameters.get('z_dim', 16)
            dropout = explainer_parameters.get('dropout', .1)
            encoder_type = explainer_parameters.get('encoder_type', 'gcn')
            disable_u = explainer_parameters.get('disable_u', False)
            lr = explainer_parameters.get('lr', 1e-3)
            weight_decay = explainer_parameters.get('weight_decay', 1e-5)
            graph_pool_type = explainer_parameters.get('graph_pool_type', 'mean')
            epochs = explainer_parameters.get('epochs', 200)
            alpha = explainer_parameters.get('alpha', 5)
            feature_dim = explainer_parameters.get('feature_dim', 2)
            
            assert feature_dim >= 2
            
            n_nodes = int(explainer_parameters['n_nodes'])
            n_labels = int(explainer_parameters['n_labels'])
            fold_id = int(explainer_parameters['fold_id'])
    
            # max_num_nodes here is equal to n_labels
            # the authors use it originally to pad the graph adjacency matrices
            # if they're different within the dataset instances.
            return self.get_clear_explainer(n_nodes, n_nodes, n_labels, batch_size_ratio,
                                            vae_type, h_dim, z_dim, dropout,
                                            encoder_type, graph_pool_type, disable_u,
                                            epochs, alpha, feature_dim, lr, weight_decay,
                                            fold_id, explainer_dict)

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
    
    def get_countergan_explainer(self, n_nodes, batch_size_ratio, device,
                                 training_iterations, n_discriminator_steps, n_generator_steps,
                                 n_labels, fold_id, ce_binarization_threshold, config_dict=None) -> Explainer:
        result = CounteRGANExplainer(self._explainer_id_counter,
                                     self._explainer_store_path,
                                     n_nodes=n_nodes,
                                     batch_size_ratio=batch_size_ratio,
                                     device=device,
                                     n_labels=n_labels,
                                     training_iterations=training_iterations,
                                     n_generator_steps=n_generator_steps,
                                     n_discriminator_steps=n_discriminator_steps,
                                     ce_binarization_threshold=ce_binarization_threshold,
                                     fold_id=fold_id, config_dict=config_dict)
        self._explainer_id_counter += 1
        return result
       
   
    def get_clear_explainer(self, n_nodes, max_num_nodes, n_labels, batch_size_ratio, vae_type,
                            h_dim, z_dim, dropout, encoder_type, graph_pool_type,
                            disable_u, epochs, alpha, feature_dim,
                            lr, weight_decay, fold_id, config_dict=None) -> Explainer:
        
        result = CLEARExplainer(self._explainer_id_counter,
                                self._explainer_store_path,
                                n_nodes=n_nodes,
                                n_labels=n_labels,
                                batch_size_ratio=batch_size_ratio,
                                vae_type=vae_type,
                                h_dim=h_dim,
                                z_dim=z_dim,
                                dropout=dropout,
                                encoder_type=encoder_type,
                                max_num_nodes=max_num_nodes,
                                graph_pool_type=graph_pool_type,
                                disable_u=disable_u,
                                epochs=epochs,
                                alpha=alpha,
                                feature_dim=feature_dim,
                                lr=lr,
                                weight_decay=weight_decay,
                                fold_id=fold_id,
                                config_dict=config_dict)
        self._explainer_id_counter += 1
        return result
        
        
        