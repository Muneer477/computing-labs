import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import duckdb
import time
import re
import json
import urllib.request
import urllib.error
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
A **community** is a group of nodes that are more strongly connected
to one another than they are to the rest of the network.

The important idea is that we do **not manually guess the groups from the picture**.
Instead, a community-detection algorithm examines the graph's connection structure
and produces a grouping of the nodes.
""")

# -------------------------------------------------
# SMALL TEACHING EXAMPLE
# -------------------------------------------------

st.subheader("First: What Does a Community Look Like?")

st.markdown("""
Before analyzing all 4,039 Facebook nodes, consider this small example.

- Nodes with the **same color** belong to the same example community.
- There are many connections **inside** each group.
- There are only a few connections **between** the groups.
""")

example_graph = nx.Graph()

left_nodes = ["A", "B", "C", "D"]
right_nodes = ["E", "F", "G", "H"]

example_graph.add_edges_from([
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "C"), ("B", "D"), ("C", "D"),
    ("E", "F"), ("E", "G"), ("E", "H"),
    ("F", "G"), ("F", "H"), ("G", "H"),
    ("D", "E")
])

example_pos = {
    "A": (0.0, 1.0),
    "B": (1.0, 1.4),
    "C": (1.0, 0.6),
    "D": (2.0, 1.0),
    "E": (4.0, 1.0),
    "F": (5.0, 1.4),
    "G": (5.0, 0.6),
    "H": (6.0, 1.0),
}

fig, ax = plt.subplots(figsize=(9, 3.5))

nx.draw_networkx_edges(
    example_graph,
    example_pos,
    width=1.5,
    alpha=0.55,
    ax=ax
)

nx.draw_networkx_nodes(
    example_graph,
    example_pos,
    nodelist=left_nodes,
    node_size=900,
    node_color=[0] * len(left_nodes),
    cmap=plt.cm.Set2,
    vmin=0,
    vmax=1,
    ax=ax
)

nx.draw_networkx_nodes(
    example_graph,
    example_pos,
    nodelist=right_nodes,
    node_size=900,
    node_color=[1] * len(right_nodes),
    cmap=plt.cm.Set2,
    vmin=0,
    vmax=1,
    ax=ax
)

nx.draw_networkx_labels(
    example_graph,
    example_pos,
    font_size=11,
    ax=ax
)

ax.text(
    1.0,
    1.82,
    "Community 1",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

ax.text(
    5.0,
    1.82,
    "Community 2",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

ax.set_title("Example: Two Densely Connected Groups with One Bridge")
ax.axis("off")
plt.tight_layout()

st.pyplot(fig, width="stretch")
plt.close(fig)

st.info("""
**Student takeaway:** Community detection tries to find this kind of grouping
automatically from the edges in the graph. We are not manually drawing circles
around nodes that simply look close together.
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
# COMMUNITY DETECTION METHODS
# -------------------------------------------------

st.subheader("Two Ways to Detect Communities")

method_col1, method_col2 = st.columns(2)

with method_col1:
    st.markdown("""
    #### Method 1: Greedy Modularity

    This method repeatedly combines groups when doing so improves
    the modularity objective.

    **Simple idea:** form groups that have relatively more connections
    inside the groups than between the groups.
    """)

with method_col2:
    st.markdown("""
    #### Method 2: Label Propagation

    Nodes begin with labels. Labels spread through neighboring nodes,
    and densely connected regions tend to settle on the same label.

    **Simple idea:** group labels emerge from the local connection
    pattern of the network.
    """)

st.caption(
    "Neither method is manually given the final number of communities."
)


@st.cache_resource
def detect_greedy_communities(path):
    graph = build_networkx_graph(path)

    communities = list(
        nx.community.greedy_modularity_communities(graph)
    )

    return sorted(
        communities,
        key=len,
        reverse=True
    )


@st.cache_resource
def detect_label_propagation_communities(path):
    graph = build_networkx_graph(path)

    communities = list(
        nx.community.asyn_lpa_communities(
            graph,
            seed=42
        )
    )

    return sorted(
        communities,
        key=len,
        reverse=True
    )


@st.cache_resource
def community_layout(path):
    graph = build_networkx_graph(path)

    return nx.spring_layout(
        graph,
        seed=42,
        iterations=20
    )


if st.button(
    "Detect and Compare Communities",
    type="primary",
    key="detect_compare_communities_button"
):
    st.session_state["community_results_ready"] = True


if st.session_state.get("community_results_ready", False):

    with st.spinner(
        "Running Greedy Modularity and Label Propagation..."
    ):

        greedy_communities = detect_greedy_communities(data_file)
        label_communities = detect_label_propagation_communities(data_file)

        greedy_modularity = nx.community.modularity(
            G,
            greedy_communities
        )

        label_modularity = nx.community.modularity(
            G,
            label_communities
        )

    st.success(
        "Both community-detection methods completed."
    )

    # -------------------------------------------------
    # METHOD COMPARISON
    # -------------------------------------------------

    st.subheader("Compare the Detection Results")

    comparison_df = pd.DataFrame([
        {
            "Method": "Greedy Modularity",
            "Communities Detected": len(greedy_communities),
            "Largest Community": len(greedy_communities[0]),
            "Modularity": greedy_modularity
        },
        {
            "Method": "Label Propagation",
            "Communities Detected": len(label_communities),
            "Largest Community": len(label_communities[0]),
            "Modularity": label_modularity
        }
    ])

    comparison_display = comparison_df.copy()

    comparison_display["Modularity"] = (
        comparison_display["Modularity"]
        .map(lambda x: f"{x:.4f}")
    )

    st.dataframe(
        comparison_display,
        width="stretch",
        hide_index=True
    )

    st.info("""
### How to read this table

Both methods analyze the **same Facebook network**, but they use different
rules for forming groups.

If the methods return different numbers of communities, that does **not**
automatically mean one method is wrong. Community detection can depend on
the rule used to define a good grouping.

The modularity value describes how strongly a particular partition separates
connections within communities from connections between communities. It should
not be interpreted as proof that one partition is the only correct social grouping.
""")

    # -------------------------------------------------
    # COMMUNITY SIZE TABLE + BETTER CHART
    # -------------------------------------------------

    st.subheader("Largest Communities")

    method_for_sizes = st.radio(
        "Choose a method to inspect",
        ["Greedy Modularity", "Label Propagation"],
        horizontal=True,
        key="community_size_method"
    )

    if method_for_sizes == "Greedy Modularity":
        selected_communities = greedy_communities
    else:
        selected_communities = label_communities

    community_rows = []

    for index, community in enumerate(
        selected_communities[:10],
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

    community_display_df = community_df.copy()

    community_display_df["Fraction of Graph"] = (
        community_display_df["Fraction of Graph"]
        .map(lambda x: f"{x:.2%}")
    )

    st.dataframe(
        community_display_df,
        width="stretch",
        hide_index=True
    )

    st.subheader("Community Size Distribution")

    st.markdown("""
Each bar represents **one detected community**.
The length of the bar shows how many Facebook nodes were assigned to it.

The communities are ordered from the **largest to the smallest**.
""")

    reversed_df = community_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        [
            f"Community {int(i)}"
            for i in reversed_df["Community"]
        ],
        reversed_df["Number of Nodes"]
    )

    ax.set_title(
        f"Top 10 Community Sizes — {method_for_sizes}"
    )

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Detected Community")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(
        bars,
        reversed_df["Number of Nodes"]
    ):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {int(value):,}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

    st.caption(
        "Community 1 is the largest detected community, "
        "Community 2 is the second largest, and so on."
    )

    # -------------------------------------------------
    # FULL-NETWORK COMMUNITY MAP
    # -------------------------------------------------

    st.subheader("See the Detected Communities in the Full Network")

    st.markdown("""
This visualization keeps the **entire Facebook graph**.

- **One dot = one Facebook node**
- **One line = one friendship edge**
- Nodes with the **same color** were assigned to the same detected community

The colors are only algorithmic labels. They do not tell us the real-world
identity of the group.
""")

    map_method = st.radio(
        "Choose the community map",
        ["Greedy Modularity", "Label Propagation"],
        horizontal=True,
        key="community_map_method"
    )

    if map_method == "Greedy Modularity":
        map_communities = greedy_communities
    else:
        map_communities = label_communities

    node_to_community = {}

    for community_index, community in enumerate(
        map_communities,
        start=1
    ):
        for node in community:
            node_to_community[node] = community_index

    node_colors = [
        node_to_community[node]
        for node in G.nodes()
    ]

    with st.spinner(
        "Preparing the full-network community map..."
    ):
        pos = community_layout(data_file)

    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_edges(
        G,
        pos,
        width=0.08,
        alpha=0.08,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=9,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.85,
        ax=ax
    )

    ax.set_title(
        f"Full Facebook Network Colored by Detected Community\n"
        f"{map_method}: {len(map_communities)} communities"
    )

    ax.axis("off")

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

    st.info("""
### How to read this community map

Look for regions where many nearby nodes share the same color.

That means the selected algorithm placed those nodes in the same detected
community based on the network's **edge structure**.

The spring layout only decides where nodes are drawn on the screen.
The community assignment itself comes from the detection algorithm.
""")

    # -------------------------------------------------
    # ZOOM INTO LARGEST COMMUNITY
    # -------------------------------------------------

    st.subheader("Zoom Into the Largest Detected Community")

    st.markdown("""
The full graph shows the overall structure. The graph below zooms into
a manageable sample of the **largest detected community** for the selected method.
""")

    largest_community = map_communities[0]

    sample_size = min(
        150,
        len(largest_community)
    )

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

        sub_pos = nx.spring_layout(
            community_subgraph,
            seed=42
        )

        nx.draw_networkx_nodes(
            community_subgraph,
            sub_pos,
            node_size=55,
            ax=ax
        )

        nx.draw_networkx_edges(
            community_subgraph,
            sub_pos,
            alpha=0.25,
            width=0.7,
            ax=ax
        )

        ax.set_title(
            f"Sample of Largest {map_method} Community "
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

    # -------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------

    st.subheader("Interpretation")

    st.markdown(
        f"""
**Greedy Modularity**

- Communities detected: **{len(greedy_communities)}**
- Largest community: **{len(greedy_communities[0]):,} nodes**
- Modularity: **{greedy_modularity:.4f}**

**Label Propagation**

- Communities detected: **{len(label_communities)}**
- Largest community: **{len(label_communities[0]):,} nodes**
- Modularity: **{label_modularity:.4f}**

The important conclusion is not that one algorithm has discovered the
"true" Facebook social groups.

Instead, both algorithms provide data-driven ways to investigate whether
the network contains densely connected groups. Comparing the methods makes
it clearer that community structure is **detected from graph connections**
rather than manually guessed from the drawing.
"""
    )

    st.warning("""
These communities are produced from graph structure only.
They should not automatically be interpreted as real-world friend groups,
schools, workplaces, or other social categories without additional evidence.
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
# JETSTREAM-HOSTED AI GRAPH LEARNING ASSISTANT
# -------------------------------------------------

st.divider()

st.header("🤖 AI Graph Learning Assistant — Jetstream + Llama 3.2")

st.markdown("""
Ask questions about the **Facebook graph and the graph-learning topics on this page**,
including degree, neighbors, paths, communities, ego networks, NetworkX,
graph representations, SQL edge tables, adjacency lists, adjacency matrices,
BFS, reachability, density, clustering, and spring layout.

The language model is **Llama 3.2 (3B)** hosted on the Jetstream virtual machine.
For questions that require exact values from this Facebook dataset, this page first
calculates the facts with **NetworkX** and then gives those verified facts to the AI
for explanation. This helps keep numerical answers grounded in the loaded graph.
""")

st.info("""
**Live Jetstream connection:** this page sends graph questions to a secure HTTPS API
hosted on the Jetstream VM. The live Streamlit app authenticates with private values
stored in **Streamlit Secrets**.

Ollama itself remains private on the Jetstream machine. Port `11434` is not exposed
directly to the public internet.
""")

st.caption(
    "Examples: “What does degree mean?”, “What is the degree of node 107?”, "
    "“Find common neighbors of 107 and 1684”, "
    "“What is the shortest path between 107 and 500?”, "
    "or “Why can community algorithms give different answers?”"
)


def _extract_node_ids(question):
    """Return number-like tokens in a student's question as strings."""
    return re.findall(r"\b\d+\b", question)


def _node_exists(graph, node_id):
    return str(node_id) in graph


def _sorted_node_list(nodes):
    nodes = list(nodes)
    try:
        return sorted(nodes, key=lambda x: int(x))
    except (TypeError, ValueError):
        return sorted(nodes)


def _format_node_list(nodes, limit=30):
    nodes = _sorted_node_list(nodes)

    if len(nodes) <= limit:
        return ", ".join(str(node) for node in nodes)

    visible = ", ".join(str(node) for node in nodes[:limit])
    return f"{visible}, ... ({len(nodes) - limit} more)"


def _is_project_related_question(question):
    """
    First code-level scope check.

    This is intentionally conservative: the assistant is for the graph-learning
    activity, not for general-purpose questions.
    """
    q = question.lower()

    project_keywords = (
        "graph", "node", "vertex", "vertices", "edge", "degree",
        "neighbor", "neighbour", "friend", "facebook", "network",
        "community", "modularity", "label propagation", "greedy",
        "ego", "path", "shortest", "hop", "reach", "reachable",
        "bfs", "breadth", "adjacency", "matrix", "list", "sql",
        "duckdb", "edge table", "networkx", "spring layout", "layout",
        "density", "clustering", "connected", "component",
        "representation", "central", "influential", "algorithm",
        "facebook_combined", "4039", "4,039", "88234", "88,234"
    )

    return any(keyword in q for keyword in project_keywords)


def build_verified_graph_context(question, graph):
    """
    Calculate exact dataset facts with NetworkX when the question appears to
    require a factual result from the loaded Facebook graph.

    The returned text is supplied to Llama as trusted context. Llama should
    explain these facts, not invent replacements for them.
    """
    original = question.strip()
    q = original.lower()
    node_ids = _extract_node_ids(original)

    facts = [
        "DATASET FACTS:",
        f"- Loaded graph nodes: {graph.number_of_nodes():,}",
        f"- Loaded graph edges: {graph.number_of_edges():,}",
        "- Graph type: undirected Facebook friendship graph",
        "- A node represents a Facebook user in this dataset.",
        "- An edge represents a friendship connection."
    ]

    added_specific_fact = False

    # Degree / direct connections
    if (
        "degree" in q
        or "direct friend" in q
        or "direct connection" in q
    ) and node_ids:
        node = node_ids[0]

        if _node_exists(graph, node):
            facts.append(
                f"- VERIFIED: degree of node {node} = {graph.degree(node):,}"
            )
        else:
            facts.append(
                f"- VERIFIED: node {node} does not exist in the loaded graph."
            )

        added_specific_fact = True

    # Ordinary neighbors
    if (
        ("neighbor" in q or "neighbour" in q or "friend of node" in q)
        and "common neighbor" not in q
        and "common neighbour" not in q
        and node_ids
    ):
        node = node_ids[0]

        if _node_exists(graph, node):
            neighbors = list(graph.neighbors(node))
            facts.append(
                f"- VERIFIED: node {node} has {len(neighbors):,} direct neighbors."
            )
            facts.append(
                f"- VERIFIED neighbor IDs for node {node}: "
                f"{_format_node_list(neighbors)}"
            )
        else:
            facts.append(
                f"- VERIFIED: node {node} does not exist in the loaded graph."
            )

        added_specific_fact = True

    # Common neighbors
    if (
        "common neighbor" in q or "common neighbour" in q
    ) and len(node_ids) >= 2:
        node_a, node_b = node_ids[0], node_ids[1]

        if not _node_exists(graph, node_a):
            facts.append(
                f"- VERIFIED: node {node_a} does not exist in the loaded graph."
            )
        elif not _node_exists(graph, node_b):
            facts.append(
                f"- VERIFIED: node {node_b} does not exist in the loaded graph."
            )
        else:
            common = list(nx.common_neighbors(graph, node_a, node_b))
            facts.append(
                f"- VERIFIED: nodes {node_a} and {node_b} have "
                f"{len(common):,} common neighbors."
            )
            facts.append(
                f"- VERIFIED common-neighbor IDs: {_format_node_list(common)}"
            )

        added_specific_fact = True

    # Shortest path / hop count
    if (
        "shortest path" in q
        or ("path" in q and len(node_ids) >= 2)
        or ("hop" in q and len(node_ids) >= 2)
    ) and len(node_ids) >= 2:
        node_a, node_b = node_ids[0], node_ids[1]

        if not _node_exists(graph, node_a):
            facts.append(
                f"- VERIFIED: node {node_a} does not exist in the loaded graph."
            )
        elif not _node_exists(graph, node_b):
            facts.append(
                f"- VERIFIED: node {node_b} does not exist in the loaded graph."
            )
        elif nx.has_path(graph, node_a, node_b):
            path = nx.shortest_path(graph, node_a, node_b)
            facts.append(
                f"- VERIFIED: shortest-path hop count from {node_a} to "
                f"{node_b} = {len(path) - 1}"
            )
            facts.append(
                f"- VERIFIED shortest path: {' -> '.join(path)}"
            )
        else:
            facts.append(
                f"- VERIFIED: there is no path between nodes {node_a} and {node_b}."
            )

        added_specific_fact = True

    # Ego network
    if "ego" in q and node_ids:
        node = node_ids[0]

        if _node_exists(graph, node):
            ego_graph_for_ai = nx.ego_graph(graph, node, radius=1)
            facts.append(
                f"- VERIFIED: radius-1 ego network of node {node} has "
                f"{ego_graph_for_ai.number_of_nodes():,} nodes and "
                f"{ego_graph_for_ai.number_of_edges():,} edges."
            )
        else:
            facts.append(
                f"- VERIFIED: node {node} does not exist in the loaded graph."
            )

        added_specific_fact = True

    # Highest degree
    if (
        "highest degree" in q
        or "largest degree" in q
        or "most connected" in q
        or "most friends" in q
    ):
        highest_node, highest_degree = max(
            graph.degree,
            key=lambda item: item[1]
        )
        facts.append(
            f"- VERIFIED: highest-degree node = {highest_node}, "
            f"degree = {highest_degree:,}"
        )
        added_specific_fact = True

    # Connected components
    if "connected component" in q or "how many components" in q:
        component_count = nx.number_connected_components(graph)
        facts.append(
            f"- VERIFIED: connected components = {component_count:,}"
        )
        added_specific_fact = True

    # Density
    if "density" in q:
        facts.append(
            f"- VERIFIED: graph density = {nx.density(graph):.10f}"
        )
        added_specific_fact = True

    # Clustering
    if (
        "average clustering" in q
        or "clustering coefficient of graph" in q
        or "overall clustering" in q
    ):
        facts.append(
            f"- VERIFIED: average clustering coefficient = "
            f"{nx.average_clustering(graph):.10f}"
        )
        added_specific_fact = True

    # Exact community results are calculated only when the student asks for them.
    if (
        "community" in q
        and (
            "how many" in q
            or "number of" in q
            or "modularity" in q
            or "largest" in q
            or "compare" in q
        )
    ):
        greedy_for_ai = detect_greedy_communities(data_file)
        label_for_ai = detect_label_propagation_communities(data_file)

        greedy_mod_for_ai = nx.community.modularity(
            graph,
            greedy_for_ai
        )
        label_mod_for_ai = nx.community.modularity(
            graph,
            label_for_ai
        )

        facts.extend([
            f"- VERIFIED Greedy Modularity communities: "
            f"{len(greedy_for_ai)}",
            f"- VERIFIED Greedy Modularity largest community: "
            f"{len(greedy_for_ai[0]):,} nodes",
            f"- VERIFIED Greedy Modularity score: "
            f"{greedy_mod_for_ai:.4f}",
            f"- VERIFIED Label Propagation communities: "
            f"{len(label_for_ai)}",
            f"- VERIFIED Label Propagation largest community: "
            f"{len(label_for_ai[0]):,} nodes",
            f"- VERIFIED Label Propagation modularity score: "
            f"{label_mod_for_ai:.4f}",
        ])
        added_specific_fact = True

    if not added_specific_fact:
        facts.append(
            "- No additional numerical calculation was required for this question."
        )

    return "\n".join(facts)


def call_jetstream_api(question, verified_context, conversation_history):
    """
    Send the student's question to the authenticated Jetstream HTTPS API.

    The API URL and bearer token are stored in Streamlit Secrets, so no private
    token is hard-coded in this source file. The Jetstream API then calls the
    locally hosted Ollama/Llama service on the VM.
    """
    try:
        api_url = st.secrets["GRAPH_AI_API_URL"].strip()
        api_token = st.secrets["GRAPH_AI_API_TOKEN"].strip()
    except Exception:
        return None, (
            "The AI service is not configured for this Streamlit deployment. "
            "GRAPH_AI_API_URL and GRAPH_AI_API_TOKEN must be present in "
            "Streamlit Secrets."
        )

    if not api_url or not api_token:
        return None, (
            "The AI service configuration is incomplete. Please check the "
            "Jetstream API URL and token in Streamlit Secrets."
        )

    recent_history = [
        message
        for message in conversation_history[-6:]
        if message.get("role") in ("user", "assistant")
        and isinstance(message.get("content"), str)
    ]

    payload = {
        "question": question,
        "verified_context": verified_context,
        "conversation_history": recent_history,
    }

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = str(result.get("answer", "")).strip()

        if not answer:
            raise RuntimeError("The Jetstream API returned an empty answer.")

        return answer, None

    except urllib.error.HTTPError as error:
        try:
            error_body = json.loads(error.read().decode("utf-8"))
            detail = error_body.get("detail", error.reason)
        except Exception:
            detail = error.reason

        return None, (
            "The Jetstream AI API rejected the request "
            f"(HTTP {error.code}: {detail})."
        )

    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        return None, (
            "I could not reach the Jetstream AI API over HTTPS. "
            f"Technical detail: {reason}"
        )

    except Exception as error:
        return None, (
            "The Jetstream AI API was reached, but the response could not "
            f"be processed. Technical detail: {error}"
        )


def answer_graph_question(question, graph, conversation_history):
    """
    Keep the assistant inside the project scope, calculate exact graph facts
    locally, and use Llama only for the natural-language explanation.
    """
    if not _is_project_related_question(question):
        return (
            "I’m designed specifically for the **Facebook Graph Investigation** "
            "on this page. Please ask me about topics such as nodes, edges, degree, "
            "neighbors, paths, communities, ego networks, NetworkX, BFS, graph "
            "representations, SQL edge tables, adjacency lists/matrices, density, "
            "clustering, or spring layout."
        ), None

    verified_context = build_verified_graph_context(
        question,
        graph
    )

    return call_jetstream_api(
        question,
        verified_context,
        conversation_history
    )


if "graph_learning_messages" not in st.session_state:
    st.session_state.graph_learning_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I’m the AI Graph Learning Assistant for this Facebook "
                "Graph Investigation. Ask me a graph-related question, and when "
                "your question needs an exact dataset value I will use NetworkX "
                "to calculate it before explaining the result."
            )
        }
    ]


# Display the saved conversation.
for message in st.session_state.graph_learning_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


with st.form(
    "jetstream_graph_learning_assistant_form",
    clear_on_submit=True
):
    student_question = st.text_input(
        "Your question",
        placeholder="Example: What is the degree of node 107?",
        key="jetstream_graph_learning_assistant_text"
    )

    ask_graph_assistant = st.form_submit_button(
        "Ask AI",
        type="primary"
    )


if ask_graph_assistant:

    if not student_question.strip():
        st.warning("Please type a graph-related question first.")

    else:
        student_question = student_question.strip()

        # Keep a copy of the conversation before adding the current question.
        previous_history = list(
            st.session_state.graph_learning_messages
        )

        st.session_state.graph_learning_messages.append(
            {
                "role": "user",
                "content": student_question
            }
        )

        with st.chat_message("user"):
            st.markdown(student_question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Calculating verified graph facts and asking Jetstream Llama..."
            ):
                assistant_answer, assistant_error = answer_graph_question(
                    student_question,
                    G,
                    previous_history
                )

            if assistant_error:
                st.error(assistant_error)

                st.info("""
The rest of the graph investigation is still available if the AI service is
temporarily unreachable. The Jetstream API, Ollama service, and HTTPS endpoint
can be checked independently without changing the graph-analysis features.
""")

                assistant_answer = (
                    "The AI connection is temporarily unavailable. "
                    "Your graph page and NetworkX calculations are still intact; "
                    "please try the question again shortly."
                )

            st.markdown(assistant_answer)

        st.session_state.graph_learning_messages.append(
            {
                "role": "assistant",
                "content": assistant_answer
            }
        )


if len(st.session_state.graph_learning_messages) > 1:
    if st.button(
        "Clear conversation",
        key="clear_jetstream_graph_learning_conversation"
    ):
        st.session_state.graph_learning_messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. Ask me another question about "
                    "the Facebook Graph Investigation."
                )
            }
        ]
        st.rerun()


st.success("""
**No paid OpenAI API is used by this assistant.**

Llama 3.2 runs on the Jetstream VM through Ollama. The Streamlit app reaches it
through the authenticated HTTPS API, while dataset-specific values are calculated
from the actual NetworkX graph before the AI explains them.
""")

st.caption(
    "The assistant is intentionally restricted to this graph-learning project. "
    "For exact dataset questions, NetworkX is the source of the factual result; "
    "Llama is used to explain the result in beginner-friendly language."
)


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