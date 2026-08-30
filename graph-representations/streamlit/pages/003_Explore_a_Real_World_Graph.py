import streamlit as st
import pandas as pd
import networkx as nx
from pathlib import Path

from graphutils import (
    load_graph,
    graph_stats
)


@st.cache_data
def cached_load_graph(data_file):
    return load_graph(data_file)


@st.cache_data
def cached_graph_stats(nodes, adj):
    return graph_stats(nodes, adj)


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("3. Exploring a Real-World Graph")

st.markdown("""
In the previous activities, we worked with a small graph that could be
drawn and inspected visually.

Real-world graphs are often much larger. In this activity, we will
explore a social-network graph from the Stanford Network Analysis Platform (SNAP).
""")

st.info("""
Dataset: Facebook Social Network (SNAP)

Nodes: 4,039  
Edges: 88,234

Source:  
https://snap.stanford.edu/data/egonets-Facebook.html
""")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

DATA_FILE = (
    Path(__file__).resolve()
    .parent.parent.parent
    / "datasets"
    / "facebook_combined.txt"
)

with open(DATA_FILE, "rb") as f:
    st.download_button(
        label="Download Dataset",
        data=f,
        file_name="facebook_combined.txt",
        mime="text/plain"
    )


edge_df, nodes, adj = cached_load_graph(DATA_FILE)

(
    degrees,
    highest_degree_node,
    highest_degree,
    avg_degree
) = cached_graph_stats(nodes, adj)

sorted_nodes = sorted(nodes, key=int)


# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("Preview of the Dataset")

st.dataframe(
    edge_df.head(20),
    hide_index=True,
    use_container_width=True
)

st.markdown("""
Each row represents an edge in the graph.

For example:

`(0, 1)`

means that nodes 0 and 1 are connected.

The graph is undirected, so the edge could equivalently be written as:

`(1, 0)`
""")


# --------------------------------------------------
# Basic Graph Statistics
# --------------------------------------------------

st.subheader("Basic Graph Statistics")

num_edges = len(edge_df)
num_nodes = len(nodes)

# Build a NetworkX graph for additional graph statistics
G = nx.Graph()

G.add_nodes_from(nodes)

G.add_edges_from(
    edge_df[["Node1", "Node2"]].itertuples(
        index=False,
        name=None
    )
)

# Graph density
density = nx.density(G)

# Average clustering coefficient
avg_clustering = nx.average_clustering(G)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col3:
    st.metric(
        "Graph Density",
        f"{density:.6f}"
    )

with col4:
    st.metric(
        "Average Clustering Coefficient",
        f"{avg_clustering:.4f}"
    )

with col1:
    st.metric(
        "Number of Nodes",
        f"{num_nodes:,}"
    )

    st.metric(
        "Number of Edges",
        f"{num_edges:,}"
    )

with col2:
    st.metric(
        "Highest Degree Node",
        highest_degree_node
    )

    st.metric(
        "Highest Degree",
        highest_degree
    )


st.info(
    f"""
Average Degree: {avg_degree:.2f}

For any undirected graph:

Average Degree = (2 × Number of Edges) / Number of Nodes

For this graph:

Average Degree = (2 × {num_edges:,}) / {num_nodes:,}

Average Degree = {avg_degree:.2f}
"""
)


# --------------------------------------------------
# Top 10 Highest-Degree Nodes
# --------------------------------------------------

st.subheader("Top 10 Highest-Degree Nodes")

top10 = sorted(
    degrees.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

top10_df = pd.DataFrame(
    top10,
    columns=["Node", "Degree"]
)

st.dataframe(
    top10_df,
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# Top 10 Degree Chart
# --------------------------------------------------

st.subheader("Top 10 Nodes by Degree")

top10_chart_df = top10_df.copy()

top10_chart_df["Node"] = (
    top10_chart_df["Node"].astype(str)
)

top10_chart_df = (
    top10_chart_df.set_index("Node")
)

st.bar_chart(
    top10_chart_df,
    use_container_width=True
)

st.info("""
This chart compares the degrees of the 10 most highly
connected nodes in the dataset.

A larger degree means that the node has more direct
connections in the social network.
""")


# --------------------------------------------------
# Degree Distribution
# --------------------------------------------------

st.subheader("Degree Distribution")

degree_values = list(
    degrees.values()
)

bin_width = 25
max_degree = max(degree_values)

distribution_rows = []

start = 0

while start <= max_degree:

    end = start + bin_width - 1

    count = sum(
        1
        for degree in degree_values
        if start <= degree <= end
    )

    # Only display ranges that actually contain nodes
    if count > 0:

        distribution_rows.append(
            {
                "Degree Range": f"{start}-{end}",
                "Number of Nodes": count,
                "Range Start": start
            }
        )

    start += bin_width


degree_distribution_df = pd.DataFrame(
    distribution_rows
)

degree_distribution_df = (
    degree_distribution_df
    .sort_values("Range Start")
)

degree_distribution_df = (
    degree_distribution_df
    .drop(columns=["Range Start"])
)


st.bar_chart(
    degree_distribution_df,
    x="Degree Range",
    y="Number of Nodes",
    use_container_width=True
)

st.info("""
The degree distribution groups nodes into ranges of 25.

For example, the 0–24 range represents nodes whose degree
is between 0 and 24.

Each bar shows how many nodes fall within that degree range.
This makes the overall connectivity pattern easier to interpret.
""")


# --------------------------------------------------
# Explore Individual Node
# --------------------------------------------------

st.subheader("Explore a Node")

selected_node = st.selectbox(
    "Choose a node",
    sorted_nodes
)

neighbors = sorted(
    adj[selected_node],
    key=int
)

degree = len(neighbors)

st.metric(
    "Degree",
    degree
)

st.info(
    f"""
Node {selected_node} is connected to {degree} other nodes.

The degree of a node is the number of neighbors that it has.
"""
)

st.markdown("### Neighbor List")

MAX_NEIGHBORS_TO_SHOW = 20

if degree <= MAX_NEIGHBORS_TO_SHOW:

    st.write(
        ", ".join(neighbors)
    )

else:

    st.write(
        ", ".join(
            neighbors[:MAX_NEIGHBORS_TO_SHOW]
        )
    )

    st.caption(
        f"Showing the first "
        f"{MAX_NEIGHBORS_TO_SHOW} of "
        f"{degree} neighbors."
    )


st.info("""
A node's degree is the number of neighbors it has.

In a social-network graph, high-degree nodes are directly
connected to many other nodes.
""")


# --------------------------------------------------
# Common Neighbors
# --------------------------------------------------

st.subheader("Common Neighbors")

col1, col2 = st.columns(2)

with col1:

    node1 = st.selectbox(
        "First Node",
        sorted_nodes,
        index=0,
        key="common_neighbors_node1_v2"
    )

with col2:

    node2 = st.selectbox(
        "Second Node",
        sorted_nodes,
        index=1,
        key="common_neighbors_node2_v2"
    )


if node1 == node2:

    st.warning(
        "Please select two different nodes "
        "to calculate common neighbors."
    )

else:

    common_neighbors = sorted(
        set(adj[node1]) &
        set(adj[node2]),
        key=int
    )

    st.metric(
        "Number of Common Neighbors",
        len(common_neighbors)
    )

    if len(common_neighbors) == 0:

        st.info(
            f"Nodes {node1} and {node2} "
            f"do not have any common neighbors."
        )

    else:

        MAX_COMMON_TO_SHOW = 20

        st.markdown(
            "### Common Neighbor List"
        )

        if len(common_neighbors) <= MAX_COMMON_TO_SHOW:

            st.write(
                ", ".join(
                    common_neighbors
                )
            )

        else:

            st.write(
                ", ".join(
                    common_neighbors[
                        :MAX_COMMON_TO_SHOW
                    ]
                )
            )

            st.caption(
                f"Showing the first "
                f"{MAX_COMMON_TO_SHOW} of "
                f"{len(common_neighbors)} "
                f"common neighbors."
            )


st.info("""
Common neighbors are nodes that are connected to both
selected nodes.

In a social-network graph, common neighbors can reveal
overlapping local connections.
""")


# --------------------------------------------------
# Connected to All
# --------------------------------------------------

st.subheader("Connected to All")

selected_nodes = st.multiselect(
    "Select 2 to 5 Nodes",
    sorted_nodes,
    default=["0", "1"],
    max_selections=5,
    key="connected_to_all_nodes_v2"
)


if len(selected_nodes) < 2:

    st.info(
        "Select at least two nodes."
    )

else:

    connected_to_all = set(
        adj[selected_nodes[0]]
    )

    for node in selected_nodes[1:]:

        connected_to_all &= set(
            adj[node]
        )

    connected_to_all = sorted(
        connected_to_all,
        key=int
    )

    st.metric(
        "Nodes Connected to All Selected Nodes",
        len(connected_to_all)
    )

    if len(connected_to_all) == 0:

        st.info(
            "No nodes are connected "
            "to all selected nodes."
        )

    else:

        MAX_TO_SHOW = 20

        st.markdown(
            "### Connected-to-All Node List"
        )

        if len(connected_to_all) <= MAX_TO_SHOW:

            st.write(
                ", ".join(
                    connected_to_all
                )
            )

        else:

            st.write(
                ", ".join(
                    connected_to_all[
                        :MAX_TO_SHOW
                    ]
                )
            )

            st.caption(
                f"Showing the first "
                f"{MAX_TO_SHOW} of "
                f"{len(connected_to_all)} nodes."
            )


    st.info("""
Nodes connected to all selected nodes can be found
by intersecting the neighbor lists of the selected nodes.

This operation will later be used to compare graph
representations and SQL-based representations.
""")
