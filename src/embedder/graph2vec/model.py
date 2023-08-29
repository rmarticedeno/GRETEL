from src.embedder.graph2vec.estimator import Graph2Vec
from src.core.trainable_base import Trainable

class Graph2VecEmbedder(Trainable):
        
    def init(self):
        self.model = Graph2Vec(**self.local_config['parameters']['model']['parameters'])
    
    def real_fit(self):
        # Copies of the graphs are provided because the _check_graphs function modifies the edges of the passed graph
        graphs = [i.graph.copy(as_view=False) for i in self.dataset.get_data()]
        print(self.model)
        self.model.fit(graphs)

    def get_embeddings(self):
        return self.model.get_embedding()

    def get_embedding(self, instance):
        # A Copy of the graph is provided because the _check_graphs function modifies the edges of the passed graph
        return self.model.infer(instance.graph.copy(as_view=False))
    
    
    
    