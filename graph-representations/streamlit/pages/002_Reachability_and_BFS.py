import streamlit as st
import graphviz
from graphutils import nodes_within_k_hops

st.set_page_config(
    page_title="Graph Representations Lab",
    layout="wide"
)

st.title(
    "2. Reachability and Breadth-First Search (BFS)"
)

st.markdown("""
Given a starting node, we often want to know which
other nodes can be reached within a small number
of hops.

We can answer this question using
**Breadth-First Search (BFS)**.
""")


# ---------------------------------------
# Toy Graph
# ---------------------------------------

edges = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("B", "D"),
    ("C", "D"),
    ("D", "E")
]

nodes = sorted(
    set([u for u, v in edges] +
        [v for u, v in edges])
)

adj = {
    node: []
    for node in nodes
}

for u, v in edges:

    adj[u].append(v)
    adj[v].append(u)


# ---------------------------------------
# Interactive BFS Demo
# ---------------------------------------

col1, col2 = st.columns([1, 1])


with col2:

    start_node = st.selectbox(
        "Starting Node",
        nodes
    )

    k = st.slider(
        "Maximum Number of Hops",
        min_value=0,
        max_value=4,
        value=0
    )

    reachable = nodes_within_k_hops(
        adj,
        start_node,
        k
    )

    st.success(
        f"Reachable Nodes: "
        f"{', '.join(sorted(reachable.keys()))}"
    )

    st.info(
        f"{len(reachable)} nodes are reachable "
        f"within {k} hops."
    )

    st.markdown("### BFS Levels")

    for distance in range(k + 1):

        nodes_at_distance = sorted(
            node
            for node, d in reachable.items()
            if d == distance
        )

        st.markdown(
            f"- **Distance {distance}:** "
            f"{', '.join(nodes_at_distance)}"
        )


# ---------------------------------------
# Graph Visualization
# ---------------------------------------

with col1:

    g = graphviz.Graph(
        format="png"
    )

    g.attr(rankdir="LR")

    g.attr(nodesep="0.3")
    g.attr(ranksep="0.4")

    g.attr(
        "node",
        shape="circle",
        fontsize="12",
        width="0.4",
        height="0.4",
        fixedsize="true"
    )

    for n in nodes:

        if n == start_node:

            g.node(
                n,
                style="filled",
                fillcolor="red",
                fontcolor="white"
            )

        elif n in reachable:

            g.node(
                n,
                style="filled",
                fillcolor="lightgreen"
            )

        else:

            g.node(n)

    for u, v in edges:

        g.edge(
            u,
            v,
            color="gray"
        )

    st.graphviz_chart(
        g,
        use_container_width=False
    )


# ---------------------------------------
# BFS Explanation
# ---------------------------------------

st.divider()

st.header(
    "How Does BFS Work?"
)

st.code(
"""
from collections import deque

# Find all nodes within k hops
#
# Input Parameters:
#   adj   : adjacency list
#   start : starting node
#   k     : maximum number of hops
#
# Returns:
#   dictionary mapping each reachable node
#   to its shortest distance from the start node

def nodes_within_k_hops(adj, start, k):

    # Starting node is at distance 0
    distances = {start: 0}

    # Queue stores (node, distance)
    queue = deque([
        (start, 0)
    ])

    while queue:

        node, dist = queue.popleft()

        # Do not explore beyond k hops
        if dist == k:
            continue

        # Explore neighbors of the current node
        for nbr in adj[node]:

            # Visit each node only once
            if nbr not in distances:

                distances[nbr] = dist + 1

                queue.append(
                    (nbr, dist + 1)
                )

    return distances
""",
language="python"
)


st.info("""
BFS explores the graph level by level.

It first explores nodes 1 hop away,
then 2 hops away,
then 3 hops away, and so on.

The `distances` dictionary records the shortest
distance from the starting node to every node
reached within the selected number of hops.
""")


st.info("""
BFS visits each node at most once
and each edge at most once.

Running Time: O(V + E)

where V is the number of nodes
and E is the number of edges.
""")