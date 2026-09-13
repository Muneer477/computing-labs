import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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
# Build NetworkX Graph
# --------------------------------------------------

G = nx.Graph()

G.add_nodes_from(nodes)

G.add_edges_from(
    edge_df[["Node1", "Node2"]].itertuples(
        index=False,
        name=None
    )
)


# --------------------------------------------------
# Facebook Friendship Network
# --------------------------------------------------

st.subheader("Facebook Friendship Network")

st.info("""
### How to Read This Network

- Each **blue dot (node)** represents one Facebook user.
- Each **line (edge)** represents a friendship between two users.
- The complete network contains **4,039 users and 88,234 friendships**.

**Why are the nodes positioned this way?**  
The graph uses a **spring layout**. Imagine the friendships as springs:
connected nodes tend to pull toward each other, while nodes also repel
each other. The algorithm uses these forces to decide where the nodes
should be placed on the screen.

The spring layout does **not change the actual network**. It only determines
where the nodes are drawn. Therefore, two nodes appearing close together
does not necessarily mean that they are directly connected. An edge between
them is what shows a direct connection.

**Why do some areas look dark or black?**  
The graph contains **88,234 edges**. In densely connected parts of the
network, many edges are drawn on top of one another. Their overlap makes
those areas appear dark or black. A dark area therefore indicates a region
with many overlapping connections; it is not a single black node.
""")

if "facebook_network_pos" not in st.session_state:
    with st.spinner("Calculating network layout..."):
        st.session_state["facebook_network_pos"] = nx.spring_layout(
            G,
            seed=42,
            iterations=20
        )

pos = st.session_state["facebook_network_pos"]

# Degree of each node
node_degrees = dict(G.degree())

# Keep every node visible, while making high-degree nodes easier to notice
node_sizes = [
    5 + (node_degrees[node] / highest_degree) * 45
    for node in G.nodes()
]

fig, ax = plt.subplots(figsize=(15, 15))

# Draw all 88,234 friendship edges
nx.draw_networkx_edges(
    G,
    pos,
    width=0.1,
    alpha=0.12,
    ax=ax
)

# Draw all 4,039 nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    alpha=0.75,
    ax=ax
)

# Identify the five highest-degree nodes
top5_nodes = [
    node
    for node, degree in sorted(
        node_degrees.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
]

# Draw the five major hubs again so they stand out
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=top5_nodes,
    node_size=[
        5 + (node_degrees[node] / highest_degree) * 90
        for node in top5_nodes
    ],
    ax=ax
)



ax.set_title(
    "Facebook Friendship Network\n"
    f"{len(nodes):,} Nodes • {len(edge_df):,} Edges"
)

ax.axis("off")

st.pyplot(fig, use_container_width=True)

plt.close(fig)


# --------------------------------------------------
# Basic Graph Statistics
# --------------------------------------------------

st.subheader("Basic Graph Statistics")

num_edges = len(edge_df)
num_nodes = len(nodes)

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

st.markdown("""
This graph ranks the **individual Facebook nodes** from the highest degree to the lower degrees.

- **Degree** = number of direct connections a node has.
- The **1st bar** is the node with the highest degree.
- The **2nd bar** is the node with the second-highest degree.
- The bars continue in **decreasing degree order** from left to right.

This is different from the degree-distribution histogram below.  
Here, **one bar = one individual node**, not a degree range.
""")

# Calculate degree for every node and sort from highest to lowest.
degrees = dict(G.degree())

top10 = sorted(
    degrees.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]

top10_df = pd.DataFrame(
    top10,
    columns=["Node", "Degree"]
)

# Add a simple rank so students can see 1st, 2nd, 3rd, etc.
top10_df.insert(
    0,
    "Rank",
    range(1, len(top10_df) + 1)
)

st.markdown("#### Ranking table")

st.dataframe(
    top10_df,
    hide_index=True,
    use_container_width=True
)

st.markdown("#### Highest degree → lowest degree")

fig, ax = plt.subplots(figsize=(11, 6))

rank_labels = [
    f"#{rank}\nNode {node}"
    for rank, node in zip(top10_df["Rank"], top10_df["Node"])
]

bars = ax.bar(
    rank_labels,
    top10_df["Degree"]
)

ax.set_title(
    "Top 10 Nodes Ranked by Degree",
    fontsize=15
)

ax.set_xlabel(
    "Rank and Node",
    fontsize=11
)

ax.set_ylabel(
    "Degree (Number of Direct Connections)",
    fontsize=11
)

ax.grid(
    axis="y",
    alpha=0.25
)

# Show the exact degree above every bar.
for bar, degree in zip(bars, top10_df["Degree"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(degree),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)

st.info("""
### How to read this graph

Start from the **left**.

The first bar is the node with the **highest degree** in the Facebook network.
The next bar has the **second-highest degree**, and so on.

For example:

**Node 107 → Degree 1045**

means Node 107 has **1,045 direct connections** in this dataset.

Because the nodes are sorted by degree, the bars should get shorter as you move
from left to right.
""")


# --------------------------------------------------
# Degree Distribution
# --------------------------------------------------

st.subheader("Degree Distribution")

degree_values = list(degrees.values())
max_degree = max(degree_values)

st.markdown("""
Before looking at the degree distribution, let's understand **degree** with a tiny example.

A **node** is one Facebook user.  
An **edge** is a friendship.  
The **degree of a node** is the number of friendship lines directly connected to it.
""")

# --------------------------------------------------
# Small teaching example
# --------------------------------------------------

example_G = nx.Graph()

example_G.add_edges_from([
    ("A", "B"),
    ("A", "C"),
    ("A", "D")
])

example_pos = {
    "A": (0, 0),
    "B": (-1, 1),
    "C": (0, 1.3),
    "D": (1, 1)
}

example_fig, example_ax = plt.subplots(figsize=(6, 3.8))

nx.draw_networkx_nodes(
    example_G,
    example_pos,
    node_size=1400,
    ax=example_ax
)

nx.draw_networkx_edges(
    example_G,
    example_pos,
    width=2,
    ax=example_ax
)

nx.draw_networkx_labels(
    example_G,
    example_pos,
    font_size=12,
    ax=example_ax
)

example_ax.set_title("Simple Degree Example")
example_ax.axis("off")

st.pyplot(example_fig, use_container_width=False)

plt.close(example_fig)

st.info("""
In this example, **Node A has degree 3** because three friendship lines
touch Node A: A–B, A–C, and A–D.

So if a Facebook user is connected to 25 other users, that user's
**degree is 25**.
""")

# --------------------------------------------------
# Student-friendly fixed-width bins
# --------------------------------------------------

st.markdown("### Now group users by degree")

st.markdown("""
Instead of using an unclear setting such as `bins=50`, we use a
**fixed bin width (span) of 20**.

That means:

- **0–19** = users with 0 to 19 connections
- **20–39** = users with 20 to 39 connections
- **40–59** = users with 40 to 59 connections
- **60–79** = users with 60 to 79 connections
- and so on

**One bar = one degree range.**  
The **height of the bar = how many users are inside that range.**
""")

bin_width = 20

# Build every 20-degree range from 0 up to the maximum degree.
bin_starts = list(range(0, max_degree + 1, bin_width))

distribution_rows = []

for range_start in bin_starts:
    range_end = range_start + bin_width - 1

    count = sum(
        1
        for degree in degree_values
        if range_start <= degree <= range_end
    )

    distribution_rows.append({
        "Range Start": range_start,
        "Range End": range_end,
        "Degree Range": f"{range_start}–{range_end}",
        "Number of Users": count
    })

degree_distribution_df = pd.DataFrame(distribution_rows)

# --------------------------------------------------
# Show the most useful part first
# --------------------------------------------------

st.markdown("### Degree Distribution")

st.caption(
    "The network goes up to degree "
    f"{max_degree}, so the graph is shown in smaller sections "
    "to keep every degree range readable."
)

range_options = []

chunk_size = 200

for chunk_start in range(0, max_degree + 1, chunk_size):
    chunk_end = min(chunk_start + chunk_size - 1, max_degree)
    range_options.append((chunk_start, chunk_end))

selected_range_label = st.selectbox(
    "Choose which degree range to view",
    [
        f"{start_value}–{end_value}"
        for start_value, end_value in range_options
    ],
    index=0,
    key="degree_distribution_range"
)

selected_index = [
    f"{start_value}–{end_value}"
    for start_value, end_value in range_options
].index(selected_range_label)

selected_start, selected_end = range_options[selected_index]

visible_df = degree_distribution_df[
    (degree_distribution_df["Range Start"] >= selected_start)
    & (degree_distribution_df["Range Start"] <= selected_end)
].copy()

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(
    visible_df["Degree Range"],
    visible_df["Number of Users"]
)

ax.set_title(
    f"Facebook Users by Degree Range ({selected_start}–{selected_end})",
    fontsize=15
)

ax.set_xlabel(
    "Degree Range (Number of Connections per User)",
    fontsize=11
)

ax.set_ylabel(
    "Number of Facebook Users",
    fontsize=11
)

ax.tick_params(
    axis="x",
    rotation=45
)

ax.grid(
    axis="y",
    alpha=0.25
)

# Put the exact number of users above each bar.
for bar, count in zip(bars, visible_df["Number of Users"]):
    if count > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

st.pyplot(fig, use_container_width=True)

plt.close(fig)

st.info(
    f"""
### How to read this graph

Look at one bar at a time.

For example, the bar labeled **20–39** represents Facebook users who have
between **20 and 39 direct connections**.

- The label at the bottom tells you the **degree range**.
- The height of the bar tells you the **number of users** in that range.
- The number written above the bar gives you the exact user count.

We use a **bin width (span) of {bin_width}**, so every bar covers exactly
{bin_width} degree values.

The complete dataset contains degree values up to **{max_degree}**.
Use the selector above the graph to look at the lower-degree users first,
then move through the higher-degree ranges without squeezing everything
into one unreadable graph.
"""
)

# --------------------------------------------------
# Log-Log Degree Distribution
# --------------------------------------------------

st.subheader("Log-Log Degree Distribution")

st.markdown("""
The histogram above grouped users into ranges such as **20–39**.

This graph does something different:

**It looks at each exact degree separately.**

For example:

- Degree **20** = a user has exactly 20 direct friendship connections.
- If **50 users** have degree 20, then the frequency of degree 20 is **50**.
""")

# Count how many users have each exact degree.
degree_frequency = {}

for degree in degree_values:
    degree_frequency[degree] = degree_frequency.get(degree, 0) + 1

degree_frequency_df = pd.DataFrame(
    sorted(degree_frequency.items()),
    columns=["Degree", "Number of Users"]
)

st.markdown("#### Step 1: Exact degree and number of users")

st.dataframe(
    degree_frequency_df.head(12),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Example: if Degree = 20 and Number of Users = 50, "
    "that means 50 different users each have exactly 20 connections."
)

st.markdown("""
So on the graph below:

- **X-axis = exact degree**
- **Y-axis = number of users with that exact degree**
""")

st.markdown("#### Step 2: Why use a log-log scale?")

st.markdown("""
The Facebook graph contains both small degree values and very large degree values.

A normal scale spaces numbers like this:

`0, 100, 200, 300, ...`

A logarithmic scale gives equal visual space to multiplication by 10:

`1, 10, 100, 1000`

This helps us see low-degree and high-degree values together more clearly.

**Important:** the log scale does not change a user's degree.  
It only changes how the numbers are spaced on the graph.
""")

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(
    degree_frequency_df["Degree"],
    degree_frequency_df["Number of Users"],
    s=24,
    alpha=0.7
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_title(
    "Log-Log Degree Distribution of the Facebook Network",
    fontsize=15
)

ax.set_xlabel(
    "Exact Degree (Number of Connections) — Log Scale",
    fontsize=11
)

ax.set_ylabel(
    "Number of Users with That Exact Degree — Log Scale",
    fontsize=11
)

ax.grid(
    True,
    which="both",
    alpha=0.25
)

plt.tight_layout()

st.pyplot(fig, use_container_width=True)

plt.close(fig)

st.info("""
### How to read one dot

Each dot represents **one exact degree value**.

Example:

If a dot represents:

- Degree = 20
- Number of Users = 50

then it means:

**50 users each have exactly 20 direct connections.**

### How to read the direction

- Move **right** → users have more connections.
- Move **up** → more users have that exact degree.
- Move **down** → fewer users have that exact degree.

### Interpreting the overall pattern

In some networks, a power-law relationship can appear approximately as a
**descending straight-line pattern** on a log-log graph.

The graph above shows the actual Facebook network data. We can examine the
observed points to see whether they approximately follow this type of pattern.
""")

st.markdown("""
#### One-sentence summary

**The histogram groups degrees into ranges, while this log-log graph shows each exact degree separately and changes the axis spacing so the overall pattern is easier to see.**
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