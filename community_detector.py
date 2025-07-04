# community_detector.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import os
import pickle
from datetime import datetime


class CommunityDetector:
    def __init__(self):
        self.graph = None
        self.communities = None
        self.metrics = {}
        self.graph_name = ""
        self.latest_result_file = 'latest_result.pkl'
        self.results_dir = 'results'
        
        # Crear directorios necesarios
        os.makedirs(self.results_dir, exist_ok=True)

    def load_karate_club_dataset(self):
        """Carga el dataset Zachary's Karate Club - un clásico en detección de comunidades"""
        self.graph = nx.karate_club_graph()
        self.graph_name = "Zachary's Karate Club"
        print(f"Dataset cargado: {self.graph_name}")
        print(f"Nodos: {self.graph.number_of_nodes()}")
        print(f"Aristas: {self.graph.number_of_edges()}")
        return self.graph

    def load_les_miserables_dataset(self):
        """Carga el dataset de Les Miserables - red de personajes"""
        self.graph = nx.les_miserables_graph()
        self.graph_name = "Les Miserables Character Network"
        print(f"Dataset cargado: {self.graph_name}")
        print(f"Nodos: {self.graph.number_of_nodes()}")
        print(f"Aristas: {self.graph.number_of_edges()}")
        return self.graph

    def create_sample_social_network(self, n_communities=3, community_size=20):
        """Crea una red social sintética con comunidades claras"""
        self.graph = nx.Graph()
        self.graph_name = "Red Social Sintética"

        node_id = 0
        for comm in range(n_communities):
            # Crear nodos de la comunidad
            nodes = list(range(node_id, node_id + community_size))

            # Conexiones internas densas
            for i in nodes:
                for j in nodes:
                    if i < j and np.random.random() < 0.3:  # 30% probabilidad de conexión interna
                        self.graph.add_edge(i, j)

            # Algunas conexiones entre comunidades
            if comm > 0:
                for _ in range(2):  # 2 conexiones entre comunidades
                    node1 = np.random.choice(nodes)
                    node2 = np.random.choice(
                        range(max(0, node_id - community_size), node_id))
                    self.graph.add_edge(node1, node2)

            node_id += community_size

        print(f"Red sintética creada: {self.graph_name}")
        print(f"Nodos: {self.graph.number_of_nodes()}")
        print(f"Aristas: {self.graph.number_of_edges()}")
        return self.graph

    def louvain_algorithm(self):
        """Implementación del algoritmo de Louvain para detección de comunidades"""
        # Usar la implementación de NetworkX
        import networkx.algorithms.community as nx_comm

        # Detectar comunidades
        communities_generator = nx_comm.louvain_communities(
            self.graph, seed=42)
        self.communities = [list(community)
                            for community in communities_generator]

        # Asignar comunidad a cada nodo
        node_to_community = {}
        for idx, community in enumerate(self.communities):
            for node in community:
                node_to_community[node] = idx

        # Guardar como atributo del grafo
        nx.set_node_attributes(self.graph, node_to_community, 'community')

        print(f"\nComunidades detectadas: {len(self.communities)}")
        for i, comm in enumerate(self.communities):
            print(f"Comunidad {i}: {len(comm)} miembros")

        return self.communities

    def girvan_newman_algorithm(self, n_communities=None):
        """Implementación del algoritmo Girvan-Newman"""
        import networkx.algorithms.community as nx_comm

        # Ejecutar el algoritmo
        comp = nx_comm.girvan_newman(self.graph)

        # Obtener el número deseado de comunidades
        if n_communities is None:
            n_communities = 3  # Por defecto

        for _ in range(n_communities - 1):
            communities_tuple = next(comp)

        self.communities = [list(community) for community in communities_tuple]

        # Asignar comunidad a cada nodo
        node_to_community = {}
        for idx, community in enumerate(self.communities):
            for node in community:
                node_to_community[node] = idx

        nx.set_node_attributes(self.graph, node_to_community, 'community')

        print(
            f"\nComunidades detectadas (Girvan-Newman): {len(self.communities)}")
        for i, comm in enumerate(self.communities):
            print(f"Comunidad {i}: {len(comm)} miembros")

        return self.communities

    def label_propagation_algorithm(self):
        """Implementación del algoritmo Label Propagation"""
        import networkx.algorithms.community as nx_comm

        # Detectar comunidades
        communities_generator = nx_comm.label_propagation_communities(
            self.graph)
        self.communities = [list(community)
                            for community in communities_generator]

        # Asignar comunidad a cada nodo
        node_to_community = {}
        for idx, community in enumerate(self.communities):
            for node in community:
                node_to_community[node] = idx

        nx.set_node_attributes(self.graph, node_to_community, 'community')

        print(
            f"\nComunidades detectadas (Label Propagation): {len(self.communities)}")
        for i, comm in enumerate(self.communities):
            print(f"Comunidad {i}: {len(comm)} miembros")

        return self.communities

    def calculate_metrics(self):
        """Calcula métricas de evaluación para las comunidades detectadas"""
        import networkx.algorithms.community as nx_comm

        if self.communities is None:
            print("Primero debes ejecutar un algoritmo de detección de comunidades")
            return

        # Modularidad
        modularity = nx_comm.modularity(self.graph, self.communities)

        # Densidad por comunidad
        densities = []
        for community in self.communities:
            subgraph = self.graph.subgraph(community)
            if len(community) > 1:
                density = nx.density(subgraph)
                densities.append(density)
            else:
                densities.append(0)

        # Coeficiente de clustering promedio
        clustering_coeffs = []
        for community in self.communities:
            coeffs = [nx.clustering(self.graph, node) for node in community]
            clustering_coeffs.append(np.mean(coeffs))

        # Grado promedio por comunidad
        avg_degrees = []
        for community in self.communities:
            degrees = [self.graph.degree(node) for node in community]
            avg_degrees.append(np.mean(degrees))

        self.metrics = {
            'modularity': round(modularity, 4),
            'num_communities': len(self.communities),
            'community_sizes': [len(c) for c in self.communities],
            'densities': [round(d, 4) for d in densities],
            'avg_clustering': [round(c, 4) for c in clustering_coeffs],
            'avg_degrees': [round(d, 2) for d in avg_degrees],
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'graph_density': round(nx.density(self.graph), 4)
        }

        print("\n=== MÉTRICAS DE EVALUACIÓN ===")
        print(f"Modularidad: {self.metrics['modularity']}")
        print(f"Número de comunidades: {self.metrics['num_communities']}")
        print(f"Tamaños de comunidades: {self.metrics['community_sizes']}")
        print(f"Densidad del grafo: {self.metrics['graph_density']}")

        return self.metrics

    def visualize_communities(self, save_path='static/current_graph.png'):
        """Visualiza el grafo con las comunidades coloreadas"""
        plt.figure(figsize=(12, 8))

        # Obtener colores para cada nodo según su comunidad
        node_colors = []
        if 'community' in nx.get_node_attributes(self.graph, 'community'):
            communities_dict = nx.get_node_attributes(self.graph, 'community')
            node_colors = [communities_dict[node]
                           for node in self.graph.nodes()]
        else:
            node_colors = 'lightblue'

        # Layout del grafo
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)

        # Dibujar el grafo
        nx.draw(self.graph, pos,
                node_color=node_colors,
                with_labels=True,
                node_size=500,
                cmap=plt.cm.rainbow,
                edge_color='gray',
                alpha=0.7)

        plt.title(f"Detección de Comunidades - {self.graph_name}", fontsize=16)
        plt.axis('off')

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(
            save_path) else '.', exist_ok=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nGráfico guardado en: {save_path}")

    def save_latest_result(self):
        """Guarda el último resultado para que los usuarios puedan verlo"""
        if self.communities is None or self.metrics is None:
            return False
        
        # Preparar datos para el frontend
        communities_data = []
        for i, community in enumerate(self.communities):
            communities_data.append({
                'id': i,
                'size': len(community),
                'members': list(community),
                'density': self.metrics['densities'][i],
                'avg_clustering': self.metrics['avg_clustering'][i],
                'avg_degree': self.metrics['avg_degrees'][i]
            })
        
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'graph_name': self.graph_name,
            'metrics': self.metrics,
            'communities': communities_data
        }
        
        try:
            # Guardar en pickle para mantener el estado completo
            with open(self.latest_result_file, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'communities': self.communities,
                    'metrics': self.metrics,
                    'graph_name': self.graph_name,
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            # También guardar en JSON para fácil lectura
            with open('latest_result.json', 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print("Resultado guardado exitosamente")
            return True
            
        except Exception as e:
            print(f"Error al guardar resultado: {e}")
            return False

    def load_latest_result(self):
        """Carga el último resultado guardado"""
        try:
            # Intentar cargar desde pickle primero
            if os.path.exists(self.latest_result_file):
                with open(self.latest_result_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Restaurar estado
                self.graph = data['graph']
                self.communities = data['communities']
                self.metrics = data['metrics']
                self.graph_name = data['graph_name']
                
                # Preparar datos para frontend
                communities_data = []
                for i, community in enumerate(self.communities):
                    communities_data.append({
                        'id': i,
                        'size': len(community),
                        'members': list(community)[:10],  # Primeros 10 miembros
                        'density': self.metrics['densities'][i],
                        'avg_clustering': self.metrics['avg_clustering'][i],
                        'avg_degree': self.metrics['avg_degrees'][i]
                    })
                
                return {
                    'timestamp': data['timestamp'],
                    'graph_name': data['graph_name'],
                    'metrics': self.metrics,
                    'communities': communities_data
                }
            
            # Si no hay pickle, intentar JSON
            elif os.path.exists('latest_result.json'):
                with open('latest_result.json', 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            print(f"Error al cargar resultado: {e}")
            return None

    def export_results(self, filename=None):
        """Exporta los resultados en formato JSON"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'results_{timestamp}.json'
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'graph_name': self.graph_name,
            'metrics': self.metrics,
            'communities': [list(comm) for comm in self.communities],
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges())
        }

        os.makedirs(self.results_dir, exist_ok=True)
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResultados exportados a: {filepath}")
        return filepath

    def get_available_results(self):
        """Obtiene lista de resultados disponibles"""
        results = []
        if os.path.exists(self.results_dir):
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.results_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            results.append({
                                'filename': filename,
                                'timestamp': data.get('timestamp', 'Unknown'),
                                'graph_name': data.get('graph_name', 'Unknown'),
                                'num_communities': data.get('metrics', {}).get('num_communities', 0)
                            })
                    except:
                        continue
        return results


# Ejemplo de uso
if __name__ == "__main__":
    # Crear detector
    detector = CommunityDetector()

    # Cargar dataset
    print("=== CARGANDO DATASET ===")
    detector.load_karate_club_dataset()

    # Ejecutar algoritmo de Louvain
    print("\n=== EJECUTANDO ALGORITMO DE LOUVAIN ===")
    detector.louvain_algorithm()

    # Calcular métricas
    detector.calculate_metrics()

    # Visualizar
    detector.visualize_communities()

    # Guardar resultado
    detector.save_latest_result()

    # Exportar resultados
    detector.export_results()

    # Probar con otro dataset
    print("\n\n=== PROBANDO CON RED SINTÉTICA ===")
    detector2 = CommunityDetector()
    detector2.create_sample_social_network(n_communities=4, community_size=15)
    detector2.louvain_algorithm()
    detector2.calculate_metrics()
    detector2.visualize_communities('static/synthetic_graph.png')
    detector2.save_latest_result()