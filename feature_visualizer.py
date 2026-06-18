from graphviz import Digraph

def create_feature_node(dot, node_id, label, color):
    """Add a feature node to the graph with consistent style."""
    dot.node(node_id, label, shape='box', style='filled', fillcolor=color)

def add_dataset_edges(dot, dataset_id, feature_ids):
    """Connect a dataset node to all feature nodes."""
    for fid in feature_ids:
        dot.edge(dataset_id, fid)

def generate_grampa_workflow(output_path='grampa_feature_workflow'):
    # Initialize directed graph
    dot = Digraph(comment='GRAMPA Feature Extraction', format='png')
    
    # ---------------- Top Level: GRAMPA Database ----------------
    dot.node(
        'GRAMPA',
        'GRAMPA Database\n(Integrated from 5 DBs)\nDBAASP, DADP, DRAMP, YADAMP, APD',
        shape='box', style='filled', fillcolor='lightgray'
    )
    
    # ---------------- Second Level: Species Datasets ----------------
    datasets = {
        'EC': 'E. coli Dataset\n(Subset from GRAMPA)',
        'SA': 'Staphylococcus aureus Dataset\n(Subset from GRAMPA)'
    }
    dataset_colors = {'EC': 'lightblue', 'SA': 'navajowhite'}
    
    for did, label in datasets.items():
        dot.node(did, label, shape='box', style='filled', fillcolor=dataset_colors[did])
        dot.edge('GRAMPA', did)  # connect GRAMPA -> dataset
    
    # ---------------- Feature Nodes ----------------
    features = {
        'SEQ': ('ProtT5 Sequence Features\nnpz', 'lightblue'),
        'STR': ('AlphaFold2 Structural Features\nPDB -> Contact Map -> CSV', 'lightgreen'),
        'PHY': ('Physicochemical Features\n32 Features, CSV', 'orange'),
        'BLO': ('BLOSUM Features\n20 Features, CSV', 'purple'),
        'POS': ('Positional Encoding Features\n32 Features, CSV', 'red')
    }
    
    for fid, (label, color) in features.items():
        create_feature_node(dot, fid, label, color)
    
    # ---------------- Connect Datasets to Features ----------------
    for did in datasets:
        add_dataset_edges(dot, did, features.keys())
    
    # ---------------- Render the Diagram ----------------
    dot.render(output_path, view=False)
    print(f"Workflow diagram saved to {output_path}.png")

# ---------------- Run the Workflow Generation ----------------
if __name__ == '__main__':
    generate_grampa_workflow('visualization/feature_extract/1')
