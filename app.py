# app.py
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import json
import os
import hashlib
from datetime import datetime
from community_detector import CommunityDetector
import networkx as nx

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Instancia global del detector
detector = CommunityDetector()

# Archivo de base de datos de usuarios
USERS_DB_FILE = './users.json'

def init_users_db():
    """Inicializa la base de datos de usuarios si no existe"""
    if not os.path.exists(USERS_DB_FILE):
        default_users = {
            "admin": {
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            },
            "user": {
                "password": hashlib.sha256("user123".encode()).hexdigest(),
                "role": "user",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
        }
        save_users_db(default_users)
        print("Base de datos de usuarios inicializada con usuarios por defecto:")
        print("Admin: admin/admin123")
        print("User: user/user123")

def load_users_db():
    """Carga la base de datos de usuarios"""
    try:
        with open(USERS_DB_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        init_users_db()
        return load_users_db()

def save_users_db(users_data):
    """Guarda la base de datos de usuarios"""
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users_data, f, indent=2)

def hash_password(password):
    """Hashea una contraseña"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    """Autentica un usuario"""
    users = load_users_db()
    if username in users:
        stored_password = users[username]['password']
        if stored_password == hash_password(password):
            # Actualizar último login
            users[username]['last_login'] = datetime.now().isoformat()
            save_users_db(users)
            return users[username]
    return None

def require_login(f):
    """Decorador para requerir login"""
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'error': 'No autenticado', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def require_admin(f):
    """Decorador para requerir rol admin"""
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({'error': 'No autenticado', 'redirect': '/login'}), 401
        if session['user']['role'] != 'admin':
            return jsonify({'error': 'No tienes permisos para esta acción'}), 403
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/')
def index():
    """Página principal - redirige según autenticación"""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login')
def login():
    """Página de login"""
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """Endpoint para autenticación"""
    username = request.json.get('username')
    password = request.json.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Usuario y contraseña son requeridos'}), 400
    
    user = authenticate_user(username, password)
    if user:
        session['user'] = {
            'username': username,
            'role': user['role'],
            'last_login': user['last_login']
        }
        return jsonify({
            'success': True,
            'user': session['user'],
            'redirect': '/'
        })
    else:
        return jsonify({'error': 'Credenciales inválidas'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Endpoint para cerrar sesión"""
    session.pop('user', None)
    return jsonify({'success': True, 'redirect': '/login'})

@app.route('/api/current_user', methods=['GET'])
@require_login
def current_user():
    """Obtiene información del usuario actual"""
    return jsonify({'user': session['user']})

@app.route('/api/load_dataset', methods=['POST'])
@require_admin
def load_dataset():
    """Carga un dataset específico - Solo admin"""
    dataset_name = request.json.get('dataset', 'karate')

    try:
        if dataset_name == 'karate':
            detector.load_karate_club_dataset()
        elif dataset_name == 'les_miserables':
            detector.load_les_miserables_dataset()
        elif dataset_name == 'synthetic':
            detector.create_sample_social_network(
                n_communities=4, community_size=15)
        else:
            return jsonify({'error': 'Dataset no válido'}), 400

        # Información básica del grafo
        graph_info = {
            'name': detector.graph_name,
            'nodes': detector.graph.number_of_nodes(),
            'edges': detector.graph.number_of_edges(),
            'density': round(nx.density(detector.graph), 4)
        }

        return jsonify({'success': True, 'graph_info': graph_info})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect_communities', methods=['POST'])
@require_admin
def detect_communities():
    """Ejecuta el algoritmo de detección de comunidades - Solo admin"""
    algorithm = request.json.get('algorithm', 'louvain')

    if detector.graph is None:
        return jsonify({'error': 'Primero debes cargar un dataset'}), 400

    try:
        if algorithm == 'louvain':
            detector.louvain_algorithm()
        elif algorithm == 'girvan_newman':
            detector.girvan_newman_algorithm(n_communities=4)
        elif algorithm == 'label_propagation':
            detector.label_propagation_algorithm()
        else:
            return jsonify({'error': 'Algoritmo no válido'}), 400

        # Calcular métricas
        detector.calculate_metrics()

        # Generar visualización
        detector.visualize_communities('static/current_graph.png')

        # Preparar datos para el frontend
        communities_data = []
        for i, community in enumerate(detector.communities):
            communities_data.append({
                'id': i,
                'size': len(community),
                'members': list(community)[:10],  # Primeros 10 miembros
                'density': detector.metrics['densities'][i],
                'avg_clustering': detector.metrics['avg_clustering'][i],
                'avg_degree': detector.metrics['avg_degrees'][i]
            })

        response_data = {
            'success': True,
            'metrics': detector.metrics,
            'communities': communities_data,
            'graph_path': '/static/current_graph.png'
        }

        # Guardar resultado para usuarios normales
        detector.save_latest_result()

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/view_latest_result', methods=['GET'])
@require_login
def view_latest_result():
    """Permite a cualquier usuario ver el último resultado"""
    try:
        result = detector.load_latest_result()
        if result:
            return jsonify({
                'success': True,
                'result': result,
                'graph_path': '/static/current_graph.png'
            })
        else:
            return jsonify({'error': 'No hay resultados disponibles'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_results', methods=['GET'])
@require_admin
def export_results():
    """Exporta los resultados actuales - Solo admin"""
    if detector.communities is None:
        return jsonify({'error': 'No hay resultados para exportar'}), 400

    try:
        filepath = detector.export_results()
        return jsonify({'success': True, 'filepath': filepath})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph_data', methods=['GET'])
@require_login
def get_graph_data():
    """Obtiene los datos del grafo en formato JSON para visualización interactiva"""
    # Intentar cargar último resultado si no hay grafo actual
    if detector.graph is None:
        result = detector.load_latest_result()
        if not result:
            return jsonify({'error': 'No hay grafo disponible'}), 400

    # Preparar datos para D3.js
    nodes = []
    for node in detector.graph.nodes():
        node_data = {
            'id': node,
            'label': str(node),
            'community': detector.graph.nodes[node].get('community', 0) if detector.communities else 0
        }
        nodes.append(node_data)

    edges = []
    for edge in detector.graph.edges():
        edges.append({
            'source': edge[0],
            'target': edge[1]
        })

    return jsonify({
        'nodes': nodes,
        'edges': edges
    })

@app.route('/api/users', methods=['GET'])
@require_admin
def get_users():
    """Obtiene lista de usuarios - Solo admin"""
    users = load_users_db()
    # Remover contraseñas de la respuesta
    safe_users = {}
    for username, user_data in users.items():
        safe_users[username] = {
            'role': user_data['role'],
            'created_at': user_data['created_at'],
            'last_login': user_data['last_login']
        }
    return jsonify({'users': safe_users})

@app.route('/api/users', methods=['POST'])
@require_admin
def create_user():
    """Crea un nuevo usuario - Solo admin"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({'error': 'Usuario y contraseña son requeridos'}), 400
    
    if role not in ['admin', 'user']:
        return jsonify({'error': 'Rol inválido'}), 400
    
    users = load_users_db()
    
    if username in users:
        return jsonify({'error': 'El usuario ya existe'}), 400
    
    users[username] = {
        'password': hash_password(password),
        'role': role,
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    
    save_users_db(users)
    
    return jsonify({'success': True, 'message': 'Usuario creado exitosamente'})

@app.route('/api/users/<username>', methods=['DELETE'])
@require_admin
def delete_user(username):
    """Elimina un usuario - Solo admin"""
    if username == session['user']['username']:
        return jsonify({'error': 'No puedes eliminar tu propia cuenta'}), 400
    
    users = load_users_db()
    
    if username not in users:
        return jsonify({'error': 'Usuario no encontrado'}), 404
    
    del users[username]
    save_users_db(users)
    
    return jsonify({'success': True, 'message': 'Usuario eliminado exitosamente'})

# Crear directorios necesarios
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Inicializar base de datos de usuarios
init_users_db()

if __name__ == '__main__':
    app.run(debug=True, port=5000)