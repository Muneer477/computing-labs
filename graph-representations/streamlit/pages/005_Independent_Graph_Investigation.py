import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import duckdb
import time
from pathlib import Path

st.set_page_config(
    page_title="Independent Graph Investigation",
    layout="wide"
)

st.title("5. Independent Graph Investigation")

st.markdown("""
In this activity, you will investigate your own
questions about the Facebook graph.
""")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    Your goal is to:

    • Formulate an interesting question.

    • Choose an appropriate representation.

    • Implement and test a solution.

    • Interpret the results.

    • Reflect on your choices.
    """)

with col2:
    st.info("""
    You may use:

    • SQL on an Edge Table

    • Python with Adjacency Lists

    • Python with Adjacency Matrices

    • AI tools or AI agents

    • Jetstream and other resources approved in class
    """)

st.warning("""
AI-generated code and AI-generated explanations
should be carefully tested and validated before
they are used.
""")

st.header("Suggested Investigation Questions")

st.markdown("""
You may investigate one or more of the following
questions, or propose a question of your own.
""")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### Neighborhoods

• Which node has the largest degree?

• How many nodes have degree greater than 100?

• How many nodes have degree equal to 1?

### Common Neighbors

• Which pair of nodes has the most common neighbors?

• Do highly connected nodes tend to share many neighbors?

### Reachability

• How many nodes are reachable within
  1, 2, 3, 4, or 5 hops?

• What fraction of the graph can be reached
  within 3 hops?
""")

with col2:

    st.markdown("""
### Communities

• Are there groups of nodes that appear to form
  tightly connected communities?

### Central Nodes

• Which nodes appear to be most influential?

• Which nodes can reach many other nodes
  within a small number of hops?

• Which nodes appear near the center of the graph?

### Your Own Question

• Use AI, class discussions, or your own
  curiosity to propose and investigate a
  question of interest.
""")

st.divider()


# -------------------------------------------------
# GETTING STARTED
# -------------------------------------------------

st.header("Community Investigation")

st.markdown("""
One possible investigation is to ask whether the Facebook graph
contains groups of nodes that are more densely connected to each
other than to the rest of the network.

Here we use NetworkX's **greedy modularity community detection**
algorithm to explore this question.
""")

# -------------------------------------------------
# FIND DATASET
# -------------------------------------------------

current_dir = Path(__file__).resolve().parent
streamlit_dir = current_dir.parent
graph_dir = streamlit_dir.parent

possible_files = [
    graph_dir / "datasets" / "facebook_combined.txt",
    streamlit_dir / "facebook_combined.txt",
    streamlit_dir / "data" / "facebook_combined.txt",
    graph_dir / "facebook_combined.txt",
]

data_file = None

for candidate in possible_files:
    if candidate.exists():
        data_file = candidate
        break

if data_file is None:
    st.error(
        "The Facebook dataset could not be found. "
        "Expected file: facebook_combined.txt"
    )
    st.stop()

# -------------------------------------------------
# LOAD GRAPH
# -------------------------------------------------

@st.cache_data
def load_facebook_edges(path):
    df = pd.read_csv(
        path,
        sep=" ",
        names=["Node1", "Node2"]
    )

    df["Node1"] = df["Node1"].astype(str)
    df["Node2"] = df["Node2"].astype(str)

    return df


@st.cache_resource
def build_networkx_graph(path):
    df = load_facebook_edges(path)

    graph = nx.Graph()

    graph.add_edges_from(
        zip(df["Node1"], df["Node2"])
    )

    return graph


edge_df = load_facebook_edges(data_file)
G = build_networkx_graph(data_file)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Nodes",
        f"{G.number_of_nodes():,}"
    )

with col2:
    st.metric(
        "Edges",
        f"{G.number_of_edges():,}"
    )

with col3:
    st.metric(
        "Connected Components",
        nx.number_connected_components(G)
    )

st.caption(
    f"Dataset loaded from: {data_file}"
)

# -------------------------------------------------
# COMMUNITY DETECTION
# -------------------------------------------------

@st.cache_resource
def detect_communities(path):
    graph = build_networkx_graph(path)

    communities = list(
        nx.community.greedy_modularity_communities(graph)
    )

    communities = sorted(
        communities,
        key=len,
        reverse=True
    )

    return communities


if st.button(
    "Detect Communities",
    type="primary",
    key="detect_communities_button"
):

    with st.spinner(
        "Detecting communities in the Facebook graph..."
    ):

        communities = detect_communities(data_file)

    st.success(
        "Community detection completed."
    )

    number_of_communities = len(communities)

    largest_community_size = len(communities[0])

    modularity_score = nx.community.modularity(
        G,
        communities
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Communities Detected",
            number_of_communities
        )

    with col2:
        st.metric(
            "Largest Community",
            f"{largest_community_size:,} nodes"
        )

    with col3:
        st.metric(
            "Modularity",
            f"{modularity_score:.4f}"
        )

    # ---------------------------------------------
    # COMMUNITY SIZE TABLE
    # ---------------------------------------------

    st.subheader("Largest Communities")

    community_rows = []

    for index, community in enumerate(
        communities[:10],
        start=1
    ):
        community_rows.append(
            {
                "Community": index,
                "Number of Nodes": len(community),
                "Fraction of Graph": (
                    len(community)
                    / G.number_of_nodes()
                )
            }
        )

    community_df = pd.DataFrame(
        community_rows
    )

    community_df["Fraction of Graph"] = (
        community_df["Fraction of Graph"]
        .map(lambda x: f"{x:.2%}")
    )

    st.dataframe(
        community_df,
        width="stretch",
        hide_index=True
    )

    # ---------------------------------------------
    # COMMUNITY SIZE BAR CHART
    # ---------------------------------------------

    st.subheader(
        "Community Size Distribution"
    )

    chart_df = pd.DataFrame(
        {
            "Community": [
                f"Community {i}"
                for i in range(
                    1,
                    min(11, len(communities) + 1)
                )
            ],
            "Number of Nodes": [
                len(c)
                for c in communities[:10]
            ]
        }
    )

    st.bar_chart(
        chart_df,
        x="Community",
        y="Number of Nodes"
    )

    # ---------------------------------------------
    # COMMUNITY VISUALIZATION
    # ---------------------------------------------

    st.subheader(
        "Visualization of the Largest Community"
    )

    st.markdown("""
Drawing all 4,039 Facebook nodes at once can make the
network difficult to interpret. Instead, the visualization
below shows a manageable sample from the largest detected
community.
""")

    largest_community = communities[0]

    sample_size = min(
        150,
        len(largest_community)
    )

    # Sort numerically when possible so the result is reproducible.
    try:
        sampled_nodes = sorted(
            largest_community,
            key=int
        )[:sample_size]
    except ValueError:
        sampled_nodes = sorted(
            largest_community
        )[:sample_size]

    community_subgraph = G.subgraph(
        sampled_nodes
    ).copy()

    if (
        community_subgraph.number_of_nodes() > 0
        and
        community_subgraph.number_of_edges() > 0
    ):

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        pos = nx.spring_layout(
            community_subgraph,
            seed=42
        )

        nx.draw_networkx_nodes(
            community_subgraph,
            pos,
            node_size=55,
            ax=ax
        )

        nx.draw_networkx_edges(
            community_subgraph,
            pos,
            alpha=0.25,
            width=0.7,
            ax=ax
        )

        ax.set_title(
            f"Sample of Largest Community "
            f"({sample_size} nodes maximum)"
        )

        ax.axis("off")

        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(fig)

    else:
        st.info(
            "The selected sample contains too few "
            "connections for a useful visualization."
        )

    # ---------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------

    st.subheader(
        "Interpretation"
    )

    st.markdown(
        f"""
The greedy modularity algorithm identified
**{number_of_communities} communities** in the graph.

The largest detected community contains
**{largest_community_size:,} nodes**.

The modularity score is **{modularity_score:.4f}**.

A larger positive modularity value suggests that
the graph contains groups whose nodes are connected
more strongly within their own groups than would be
expected from a comparable random network.

This supports investigating the Facebook network as
a graph with meaningful community structure rather
than treating all nodes as one uniform group.
"""
    )

    st.info("""
These communities are produced by an algorithm.
They should not automatically be interpreted as
real-world social groups without additional evidence.
""")

st.divider()
# -------------------------------------------------
# EGO NETWORK INVESTIGATION
# -------------------------------------------------

st.header("Ego Network Investigation")

st.markdown("""
Another way to investigate a social network is to examine the
**ego network** of an individual node.

An ego network contains:

- A selected central node, called the **ego**
- The ego's direct neighbors
- The connections among those neighbors

This provides a local view of the Facebook graph without drawing
all 4,039 nodes at once.
""")

# Sort node IDs numerically
try:
    ego_node_options = sorted(
        G.nodes(),
        key=int
    )
except ValueError:
    ego_node_options = sorted(
        G.nodes()
    )

# Default to node 0 when it exists
if "0" in ego_node_options:
    default_ego_index = ego_node_options.index("0")
else:
    default_ego_index = 0

selected_ego_node = st.selectbox(
    "Choose an ego node",
    ego_node_options,
    index=default_ego_index,
    key="ego_network_node_v2"
)

show_ego_labels = st.checkbox(
    "Show node labels",
    value=False,
    key="ego_network_labels"
)

# Build the 1-hop ego network
ego_graph = nx.ego_graph(
    G,
    selected_ego_node,
    radius=1
)

ego_degree = G.degree(
    selected_ego_node
)

ego_neighbors = list(
    G.neighbors(selected_ego_node)
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Ego Node Degree",
        f"{ego_degree:,}"
    )

with col2:
    st.metric(
        "Nodes in Ego Network",
        f"{ego_graph.number_of_nodes():,}"
    )

with col3:
    st.metric(
        "Edges in Ego Network",
        f"{ego_graph.number_of_edges():,}"
    )

st.info(
    f"""
Node {selected_ego_node} has **{ego_degree:,} direct neighbors**.

Its 1-hop ego network contains the selected node,
all of its direct neighbors, and the connections
that exist among those neighbors.
"""
)

# -------------------------------------------------
# EGO NETWORK VISUALIZATION
# -------------------------------------------------

st.subheader(
    f"1-Hop Ego Network of Node {selected_ego_node}"
)

st.markdown("""
The visualization below reproduces the type of ego-network
analysis used in the Colab investigation, but allows a different
node to be selected interactively.
""")

# Large ego networks can become visually crowded.
# Limit the drawing while keeping the numerical statistics above exact.
MAX_EGO_NODES_TO_DRAW = 150

if ego_graph.number_of_nodes() <= MAX_EGO_NODES_TO_DRAW:

    ego_graph_to_draw = ego_graph.copy()
    visualization_note = None

else:

    # Keep the ego node and a deterministic sample of its neighbors.
    try:
        sorted_ego_neighbors = sorted(
            ego_neighbors,
            key=int
        )
    except ValueError:
        sorted_ego_neighbors = sorted(
            ego_neighbors
        )

    sampled_neighbors = sorted_ego_neighbors[
        :MAX_EGO_NODES_TO_DRAW - 1
    ]

    nodes_to_draw = [
        selected_ego_node
    ] + sampled_neighbors

    ego_graph_to_draw = ego_graph.subgraph(
        nodes_to_draw
    ).copy()

    visualization_note = (
        f"The complete ego network contains "
        f"{ego_graph.number_of_nodes():,} nodes. "
        f"For readability, the visualization shows "
        f"{ego_graph_to_draw.number_of_nodes():,} nodes. "
        f"The statistics above still describe the complete "
        f"1-hop ego network."
    )

if visualization_note:
    st.warning(
        visualization_note
    )

if ego_graph_to_draw.number_of_nodes() > 0:

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    pos = nx.spring_layout(
        ego_graph_to_draw,
        seed=42
    )

    # Draw all nodes first
    nx.draw_networkx_nodes(
        ego_graph_to_draw,
        pos,
        node_size=45,
        ax=ax
    )

    # Draw the selected ego node separately
    # with a larger size so it is easy to identify.
    if selected_ego_node in ego_graph_to_draw:

        nx.draw_networkx_nodes(
            ego_graph_to_draw,
            pos,
            nodelist=[selected_ego_node],
            node_size=220,
            ax=ax
        )

    nx.draw_networkx_edges(
        ego_graph_to_draw,
        pos,
        alpha=0.25,
        width=0.6,
        ax=ax
    )

    if show_ego_labels:

        nx.draw_networkx_labels(
            ego_graph_to_draw,
            pos,
            font_size=7,
            ax=ax
        )

    ax.set_title(
        f"1-Hop Ego Network of Node "
        f"{selected_ego_node}"
    )

    ax.axis("off")

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

else:

    st.info(
        "No nodes are available for this ego-network visualization."
    )

# -------------------------------------------------
# EGO NETWORK INTERPRETATION
# -------------------------------------------------

st.subheader("Interpretation")

st.markdown(
    f"""
For node **{selected_ego_node}**:

- Degree = **{ego_degree:,}**
- Direct neighbors = **{len(ego_neighbors):,}**
- Nodes in the 1-hop ego network = **{ego_graph.number_of_nodes():,}**
- Edges in the 1-hop ego network = **{ego_graph.number_of_edges():,}**

The ego network gives a local view of how the selected node
is embedded in the larger Facebook graph.

A dense ego network indicates that many of the selected node's
neighbors are also connected to one another, while a sparse ego
network indicates fewer connections among those neighbors.
"""
)

st.info("""
The ego-network visualization describes graph structure only.
The dataset does not provide enough information here to infer
the real-world identities or relationships represented by
individual node IDs.
""")

st.divider()

# -------------------------------------------------
# GRAPH REPRESENTATION LABORATORY
# -------------------------------------------------

st.header("🧪 Graph Representation Laboratory")

st.markdown("""
This laboratory runs the **same neighbor query using three graph representations**:

- **Adjacency List**
- **Adjacency Matrix**
- **SQL Edge Table**

It checks both **correctness** and **performance**. The three custom
implementations are also checked against **NetworkX as an independent reference**
before their computational costs are compared.
""")

@st.cache_resource
def build_lab_representations(path):
    import numpy as np

    df = load_facebook_edges(path)
    node_set = set(df["Node1"]) | set(df["Node2"])

    try:
        sorted_nodes = sorted(node_set, key=int)
    except ValueError:
        sorted_nodes = sorted(node_set)

    adjacency_list = {node: set() for node in sorted_nodes}

    for u, v in zip(df["Node1"], df["Node2"]):
        adjacency_list[u].add(v)
        adjacency_list[v].add(u)

    node_to_index = {node: i for i, node in enumerate(sorted_nodes)}
    adjacency_matrix = np.zeros(
        (len(sorted_nodes), len(sorted_nodes)),
        dtype=bool
    )

    for u, v in zip(df["Node1"], df["Node2"]):
        i = node_to_index[u]
        j = node_to_index[v]
        adjacency_matrix[i, j] = True
        adjacency_matrix[j, i] = True

    return sorted_nodes, adjacency_list, adjacency_matrix, node_to_index


@st.cache_resource
def build_lab_sql_connection(path):
    df = load_facebook_edges(path)
    conn = duckdb.connect(database=":memory:")
    conn.register("facebook_edges_df", df)
    conn.execute("""
        CREATE TABLE EdgeTable AS
        SELECT
            CAST(Node1 AS VARCHAR) AS node1,
            CAST(Node2 AS VARCHAR) AS node2
        FROM facebook_edges_df
    """)
    return conn


lab_nodes, lab_adj, lab_matrix, lab_node_to_index = build_lab_representations(data_file)
lab_conn = build_lab_sql_connection(data_file)


def lab_neighbors_adjacency_list(node):
    return set(lab_adj[node])


def lab_neighbors_adjacency_matrix(node):
    row_index = lab_node_to_index[node]
    neighbor_indices = lab_matrix[row_index].nonzero()[0]
    return {lab_nodes[index] for index in neighbor_indices}


def lab_neighbors_sql(node):
    result = lab_conn.execute(
        """
        SELECT node2 AS neighbor
        FROM EdgeTable
        WHERE node1 = ?
        UNION
        SELECT node1 AS neighbor
        FROM EdgeTable
        WHERE node2 = ?
        """,
        [node, node]
    ).fetchall()

    return {str(row[0]) for row in result}


def measure_lab_operation(function, *args, repetitions=10):
    times = []
    result = None

    function(*args)  # warm-up

    for _ in range(repetitions):
        start_time = time.perf_counter()
        result = function(*args)
        times.append(time.perf_counter() - start_time)

    times.sort()
    return result, times[len(times) // 2]


st.subheader("Experiment 1: Neighbor Query")

st.markdown("""
Choose a node and run the same neighbor query using all three representations.
The laboratory first validates the answers and then compares median execution
time over **10 measured runs** after one warm-up run.
""")

lab_default_index = lab_nodes.index("107") if "107" in lab_nodes else 0

lab_selected_node = st.selectbox(
    "Choose a node for the representation experiment",
    lab_nodes,
    index=lab_default_index,
    key="representation_lab_neighbor_node"
)

st.caption(
    "Node 107 is the default because it is the highest-degree node "
    "in this Facebook dataset."
)

if st.button(
    "Run Representation Race",
    type="primary",
    key="run_representation_race"
):
    with st.spinner("Running and validating the three representations..."):
        list_result, list_time = measure_lab_operation(
            lab_neighbors_adjacency_list, lab_selected_node
        )
        matrix_result, matrix_time = measure_lab_operation(
            lab_neighbors_adjacency_matrix, lab_selected_node
        )
        sql_result, sql_time = measure_lab_operation(
            lab_neighbors_sql, lab_selected_node
        )

    # Independent reference answer using NetworkX.
    networkx_result = set(G.neighbors(lab_selected_node))

    four_way_agree = (
        networkx_result
        == list_result
        == matrix_result
        == sql_result
    )

    st.subheader("Independent Correctness Validation")

    st.markdown("""
The three custom implementations are checked against **NetworkX** as an
independent reference. This is stronger than only checking whether the
three implementations agree with one another.
""")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("NetworkX Reference", f"{len(networkx_result):,} neighbors")
    with col2:
        st.metric("Adjacency List", f"{len(list_result):,} neighbors")
    with col3:
        st.metric("Adjacency Matrix", f"{len(matrix_result):,} neighbors")
    with col4:
        st.metric("SQL Edge Table", f"{len(sql_result):,} neighbors")

    if four_way_agree:
        st.success(
            "✅ 4-WAY VALIDATION PASSED — NetworkX, adjacency list, "
            "adjacency matrix, and SQL returned exactly the same neighbor set."
        )
    else:
        st.error(
            "⚠️ VALIDATION FAILED — at least one custom representation "
            "does not match the independent NetworkX reference."
        )

        validation_df = pd.DataFrame({
            "Implementation": [
                "Adjacency List",
                "Adjacency Matrix",
                "SQL Edge Table"
            ],
            "Matches NetworkX": [
                list_result == networkx_result,
                matrix_result == networkx_result,
                sql_result == networkx_result
            ]
        })

        st.dataframe(validation_df, width="stretch", hide_index=True)

    st.subheader("Representation Race")

    timing_df = pd.DataFrame({
        "Representation": [
            "Adjacency List",
            "Adjacency Matrix",
            "SQL Edge Table"
        ],
        "Median Time (seconds)": [
            list_time,
            matrix_time,
            sql_time
        ]
    }).sort_values("Median Time (seconds)").reset_index(drop=True)

    timing_df["Rank"] = timing_df.index + 1

    display_df = timing_df[
        ["Rank", "Representation", "Median Time (seconds)"]
    ].copy()

    display_df["Median Time (seconds)"] = display_df[
        "Median Time (seconds)"
    ].map(lambda value: f"{value:.9f}")

    st.dataframe(display_df, width="stretch", hide_index=True)

    st.bar_chart(
        timing_df.set_index("Representation")[["Median Time (seconds)"]]
    )

    fastest_name = timing_df.iloc[0]["Representation"]
    fastest_time = timing_df.iloc[0]["Median Time (seconds)"]

    st.success(f"🏆 Fastest in this local run: **{fastest_name}**")

    relative_rows = []
    for _, row in timing_df.iterrows():
        ratio = row["Median Time (seconds)"] / fastest_time
        relative_rows.append({
            "Representation": row["Representation"],
            "Relative to Fastest": (
                "1.0× (fastest)" if ratio == 1
                else f"{ratio:,.1f}× slower"
            )
        })

    st.dataframe(
        pd.DataFrame(relative_rows),
        width="stretch",
        hide_index=True
    )

    st.info("""
These timings are a **local interactive experiment**, not a
hardware-independent benchmark. Small operations can be affected by Python
overhead, caching, operating-system scheduling, and the local machine.

The strongest evidence comes from combining **independent reference validation**,
**cross-representation correctness**, and controlled experiments such as the
Jetstream benchmarks.
""")

    with st.expander("Show validated neighbor IDs"):
        try:
            displayed_neighbors = sorted(list_result, key=int)
        except ValueError:
            displayed_neighbors = sorted(list_result)

        st.write(displayed_neighbors)

st.divider()

# -------------------------------------------------
# GETTING STARTED
# -------------------------------------------------

st.header("Getting Started")

col1, col2 = st.columns(2)

with col1:

    st.subheader("SQL")

    st.markdown("""
Store the graph as an Edge Table.

Each row contains the endpoints of one edge.
""")

    st.code(
"""
CREATE TABLE EdgeTable
(
    node1 INTEGER,
    node2 INTEGER
);
""",
        language="sql"
    )

    st.markdown("""
Import the Facebook dataset into
the EdgeTable table.

You can then write SQL queries to investigate
your question.
""")

with col2:

    st.subheader("Python")

    st.markdown("""
Represent the graph using an adjacency list.
""")

    st.code(
"""
import pandas as pd

# Load the Facebook graph and store it as an edge list.
edge_df = pd.read_csv(
    "facebook_combined.txt",
    sep=" ",
    names=["node1", "node2"]
)

edges = list(
    zip(
        edge_df["node1"],
        edge_df["node2"]
    )
)

# Build the adjacency list.
adj = {}

for u, v in edges:
    adj.setdefault(u, []).append(v)
    adj.setdefault(v, []).append(u)
""",
        language="python"
    )

    st.markdown("""
In this representation:

• adj[x] contains the neighbors of x

• The neighbors may not be sorted

• Use sorted(adj[x]) if needed
""")

# -------------------------------------------------
# ADDITIONAL PYTHON TOOLS
# -------------------------------------------------

st.header("Additional Python Tools")

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Breadth-First Search (BFS)"
    )

    st.code(
"""
from collections import deque

def nodes_within_k_hops(
    adj,
    start,
    k
):
    distances = {start: 0}

    queue = deque(
        [(start, 0)]
    )

    while queue:

        node, dist = queue.popleft()

        if dist == k:
            continue

        for nbr in adj[node]:

            if nbr not in distances:

                distances[nbr] = dist + 1

                queue.append(
                    (nbr, dist + 1)
                )

    return distances
""",
        language="python"
    )

with col2:

    st.subheader(
        "Adjacency Matrix"
    )

    st.code(
"""
# Assume edges and adj have already been computed.

nodes = set(
    [u for u, v in edges]
    +
    [v for u, v in edges]
)

matrix = {}

for u in nodes:

    matrix[u] = {
        v: 0
        for v in nodes
    }

    for nbr in adj[u]:

        matrix[u][nbr] = 1
""",
        language="python"
    )

st.info("""
The code snippets above are intended as starting
points. You may modify them, combine them, or use
AI tools to generate alternative implementations.

Be sure to test and validate your results.
""")

# -------------------------------------------------
# USING AI
# -------------------------------------------------

st.header("Using AI")

st.markdown("""
AI tools may help you:

• Formulate a question

• Design an algorithm

• Generate SQL

• Generate Python code

• Interpret results

Remember to validate all AI-generated
code and explanations.
""")

# -------------------------------------------------
# USING JETSTREAM
# -------------------------------------------------

st.header("Using Jetstream")

st.markdown("""
If you would like to run your own Python
programs using Jetstream or other resources
approved in class, please follow the
instructions provided in class.

Jetstream may be useful for:

• Running larger experiments

• Evaluating AI-generated code

• Comparing graph representations

• Measuring performance
""")

st.success("""
You are now ready to investigate your own
questions about the Facebook graph.

Choose a question, select an appropriate
representation, implement a solution,
evaluate the results, and reflect on your
choices.

Good investigations often begin with
simple questions and evolve into more
interesting ones.
""")