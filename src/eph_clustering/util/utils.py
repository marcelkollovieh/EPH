from heapdict import *
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sknetwork.hierarchy import tree_sampling_divergence
from eph_clustering.algorithms.agglomerative_linkage import agglomerative_linkage


# Get size of clusters in the hierarchy.
def get_cluster_sizes(T, n_leaves):
    def get_cluster_size(T, node):
        size = 0
        for child in T[node]:
            if child < n_leaves:
                size += 1
            else:
                size += get_cluster_size(T, child)
        return size

    sizes = {}
    for parent in T.keys():
        sizes[parent] = get_cluster_size(T, parent)
    return sizes


# Samples a tree from the learned parent probabilities.
def sample_tree(M1, M2):
    n_leaves, n_internal = M1.shape
    parents = {}

    tree = {k: [] for k in range(n_leaves, n_leaves + n_internal)}

    # Sample tree.
    for leaf in range(M1.shape[0]):
        parent = np.random.choice(M1.shape[1], p=M1[leaf, :]) + M1.shape[0]
        tree[parent].append(leaf)
        parents[leaf] = parent

    for internal in range(M2.shape[0] - 1):
        parent = np.random.choice(M2.shape[1], p=M2[internal, :]) + M1.shape[0]
        tree[parent].append(internal + M1.shape[0])
        parents[internal + M1.shape[0]] = parent

    # Prune tree.
    updated = True
    while updated:
        updated = False
        for k, v in tree.items():
            # Do not prune out the root node.
            if k == n_leaves + n_internal - 1:
                continue

            # Prune out node if it has none or only one child.
            if len(v) <= 1:
                assert k in parents
                parent = parents[k]
                if len(v) == 1:
                    # Add child to parent.
                    tree[parent].append(v[0])
                    parents[v[0]] = parent

                # Remove node from children of parent.
                tree[parent] = [n for n in tree[parent] if n != k]
                # Remove node from tree.
                del parents[k]
                del tree[k]
                updated = True
                break

    # Special handling of the root node.
    if len(tree[n_leaves + n_internal - 1]) == 1:
        child = tree[n_leaves + n_internal - 1][0]
        tree[n_leaves + n_internal - 1] = tree[child]
        del tree[child]
        del parents[child]
        for child in tree[n_leaves + n_internal - 1]:
            parents[child] = n_leaves + n_internal - 1

    return tree


def tree_to_A_B(T, num_nodes, num_internal_nodes, dtype=torch.float32):
    T_tmp = {}
    label_mapping = {
        internal_label: num_nodes + new_internal_label
        for new_internal_label, internal_label in enumerate(sorted(list(T.keys())))
    }
    for internal_label, new_internal_label in label_mapping.items():
        T_tmp[new_internal_label] = []
        for child in T[internal_label]:
            if child < num_nodes:
                T_tmp[new_internal_label].append(child)
            else:
                T_tmp[new_internal_label].append(label_mapping[child])

    if isinstance(dtype, np.dtype):
        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        elif dtype == np.int32:
            dtype = torch.int32
        elif dtype == np.int64:
            dtype = torch.int64
        else:
            raise NotImplementedError(f"unknown dtype {dtype}")
    # A and B
    A_tree = torch.zeros((num_nodes, num_internal_nodes), dtype=dtype)
    B_tree = torch.zeros((num_internal_nodes, num_internal_nodes), dtype=dtype)
    for internal_label, children in T_tmp.items():
        j = int(internal_label - num_nodes)
        for child in children:
            if child < num_nodes:
                i = int(child)
                A_tree[i, j] = 1
            else:
                i = int(child - num_nodes)
                B_tree[i, j] = 1
    return A_tree, B_tree


def dendrogram_to_tree_b(dendrogram, n_leaves):
    tree, d = {}, {u: float("inf") for u in range(n_leaves)}
    for t in range(n_leaves - 1):
        u = dendrogram[t, 0]
        v = dendrogram[t, 1]
        dist = dendrogram[t, 2]
        if dist == d[u] and dist == d[v]:
            tree[n_leaves + t] = tree.pop(u)
            tree[n_leaves + t] += tree.pop(v)
        elif dist == d[u]:
            tree[n_leaves + t] = [v]
            tree[n_leaves + t] += tree.pop(u)
        elif dist == d[v]:
            tree[n_leaves + t] = [u]
            tree[n_leaves + t] += tree.pop(v)
        else:
            tree[n_leaves + t] = [u, v]
        d[n_leaves + t] = dist
    return tree


class ClusterMultTree:
    def __init__(self, cluster_label, distance, pi, sum_p_ab, sum_pi_a_pi_b):
        self.cluster_label = cluster_label
        self.merged_clusters = set([cluster_label])
        self.d_ab = distance
        self.pi = pi
        self.sum_pi_a_pi_b = sum_pi_a_pi_b
        self.sum_p_ab = sum_p_ab
        self.up_merge_loss = float("inf")
        self.up_merge_d_ab = None
        self.father = None
        self.children = set([])


def xlogy(x, y):
    if x == 0.0:
        return 0.0
    else:
        return x * np.log(y)


def compress_hierarchy_tsd(graph, dendrogram, n_internal):
    graph_copy = graph.copy()
    n_nodes = np.shape(dendrogram)[0] + 1
    assert n_nodes > n_internal

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for u, v in graph_copy.edges():
        weight = graph_copy[u][v]["weight"]
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight

    # Build the ClusterMultTree
    u = n_nodes
    cluster_trees = {
        t: ClusterMultTree(t, None, w[t] / float(wtot), 0.0, 0.0)
        for t in range(n_nodes)
    }
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        # Building of the new level
        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]
        pi_a = left_tree.pi
        pi_b = right_tree.pi

        w[u] = w.pop(a) + w.pop(b)
        pi = w[u] / float(wtot)
        if graph_copy.has_edge(a, b):
            p_ab = 2 * graph_copy[a][b]["weight"] / float(wtot)
        else:
            p_ab = 0
        d_ab = p_ab / float(pi_a * pi_b)
        new_tree = ClusterMultTree(u, d_ab, pi, p_ab, pi_a * pi_b)
        new_tree.children.add(left_tree)
        left_tree.father = new_tree
        new_tree.children.add(right_tree)
        right_tree.father = new_tree
        cluster_trees[u] = new_tree

        # Update graph
        graph_copy.add_node(u)
        neighbors_a = list(graph_copy.neighbors(a))
        neighbors_b = list(graph_copy.neighbors(b))
        for v in neighbors_a:
            graph_copy.add_edge(u, v, weight=graph_copy[a][v]["weight"])
        for v in neighbors_b:
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]["weight"] += graph_copy[b][v]["weight"]
            else:
                graph_copy.add_edge(u, v, weight=graph_copy[b][v]["weight"])
        graph_copy.remove_node(a)
        graph_copy.remove_node(b)

        u += 1

    # Compute the information loss of each possible merge
    u = 2 * n_nodes - 2
    merging_priority = heapdict()
    for t in list(reversed(range(n_nodes - 1))):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]

        current_tree = cluster_trees[u]
        d_ab = current_tree.d_ab
        p_ab = current_tree.sum_p_ab
        pi_a_pi_b = current_tree.sum_pi_a_pi_b

        # Loss computation with left level
        if left_tree.d_ab is not None:
            left_d_ab = left_tree.d_ab
            left_p_ab = left_tree.sum_p_ab
            left_pi_a_pi_b = left_tree.sum_pi_a_pi_b

            left_tree.up_merge_d_ab = (p_ab + left_p_ab) / float(
                pi_a_pi_b + left_pi_a_pi_b
            )
            left_tree.up_merge_loss = xlogy(
                p_ab + left_p_ab, left_tree.up_merge_d_ab
            ) - (xlogy(p_ab, d_ab) + xlogy(left_p_ab, left_d_ab))
            left_tree.up_merge_loss = -left_tree.up_merge_loss

            merging_priority[left_tree.cluster_label] = left_tree.up_merge_loss

        # Loss computation with right level
        if right_tree.d_ab is not None:
            right_d_ab = right_tree.d_ab
            right_p_ab = right_tree.sum_p_ab
            right_pi_a_pi_b = right_tree.sum_pi_a_pi_b

            right_tree.up_merge_d_ab = (p_ab + right_p_ab) / float(
                (pi_a_pi_b + right_pi_a_pi_b)
            )
            right_tree.up_merge_loss = xlogy(
                p_ab + right_p_ab, right_tree.up_merge_d_ab
            ) - (xlogy(p_ab, d_ab) + xlogy(right_p_ab, right_d_ab))
            right_tree.up_merge_loss = -right_tree.up_merge_loss

            merging_priority[right_tree.cluster_label] = right_tree.up_merge_loss

        u -= 1

    # Merge n_levels times
    while len(merging_priority) >= n_internal:
        cluster_label, minimum_loss = merging_priority.popitem()

        merged_tree = cluster_trees[cluster_label]
        father_merged_tree = merged_tree.father

        # Merge the two levels
        father_merged_tree.sum_pi_a_pi_b += merged_tree.sum_pi_a_pi_b
        father_merged_tree.sum_p_ab += merged_tree.sum_p_ab
        father_merged_tree.d_ab = merged_tree.up_merge_d_ab
        father_merged_tree.children |= merged_tree.children
        father_merged_tree.children.remove(merged_tree)
        cluster_trees.pop(cluster_label)

        # Updates the father and the children loss
        if father_merged_tree.father is not None:
            pi_a_pi_b = father_merged_tree.sum_pi_a_pi_b
            p_ab = father_merged_tree.sum_p_ab
            d_ab = father_merged_tree.d_ab
            father_pi_a_pi_b = father_merged_tree.father.sum_pi_a_pi_b
            father_p_ab = father_merged_tree.father.sum_p_ab
            father_d_ab = father_merged_tree.father.d_ab

            father_merged_tree.up_merge_d_ab = (p_ab + father_p_ab) / float(
                (pi_a_pi_b + father_pi_a_pi_b)
            )
            father_merged_tree.up_merge_loss = xlogy(
                p_ab + father_p_ab, father_merged_tree.up_merge_d_ab
            ) - (xlogy(p_ab, d_ab) + xlogy(father_p_ab, father_d_ab))
            father_merged_tree.up_merge_loss = -father_merged_tree.up_merge_loss

            merging_priority[
                father_merged_tree.cluster_label
            ] = father_merged_tree.up_merge_loss

        for child in father_merged_tree.children:
            if child.d_ab is not None:
                pi_a_pi_b = father_merged_tree.sum_pi_a_pi_b
                p_ab = father_merged_tree.sum_p_ab
                d_ab = father_merged_tree.d_ab
                child_pi_a_pi_b = child.sum_pi_a_pi_b
                child_p_ab = child.sum_p_ab
                child_d_ab = child.d_ab

                child.up_merge_d_ab = (p_ab + child_p_ab) / float(
                    (pi_a_pi_b + child_pi_a_pi_b)
                )
                child.up_merge_loss = xlogy(p_ab + child_p_ab, child.up_merge_d_ab) - (
                    xlogy(p_ab, d_ab) + xlogy(child_p_ab, child_d_ab)
                )
                child.up_merge_loss = -child.up_merge_loss

                child.father = father_merged_tree

                merging_priority[child.cluster_label] = child.up_merge_loss

    keys = list(cluster_trees.keys())
    for t in keys:
        if len(cluster_trees[t].children) == 0:
            cluster_trees.pop(t)
    compressed_tree = {
        c.cluster_label: [child.cluster_label for child in c.children]
        for i, c in cluster_trees.items()
    }
    return compressed_tree


def compress_hierarchy_dasgupta(graph, dendrogram, n_internal):
    graph_copy = graph.copy()
    n_nodes = np.shape(dendrogram)[0] + 1
    assert n_nodes > n_internal

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for u, v in graph_copy.edges():
        weight = graph_copy[u][v]["weight"]
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight

    # Build the ClusterMultTree
    u = n_nodes
    cluster_trees = {
        t: ClusterMultTree(t, None, w[t] / float(wtot), 0.0, 0.0)
        for t in range(n_nodes)
    }
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        # Building of the new level
        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]
        pi_a = left_tree.pi
        pi_b = right_tree.pi

        w[u] = w.pop(a) + w.pop(b)
        pi = w[u] / float(wtot)
        if graph_copy.has_edge(a, b):
            p_ab = 2 * graph_copy[a][b]["weight"] / float(wtot)
        else:
            p_ab = 0
        d_ab = p_ab * float(pi_a + pi_b)
        new_tree = ClusterMultTree(u, d_ab, pi, p_ab, pi_a + pi_b)
        new_tree.children.add(left_tree)
        left_tree.father = new_tree
        new_tree.children.add(right_tree)
        right_tree.father = new_tree
        cluster_trees[u] = new_tree

        # Update graph
        graph_copy.add_node(u)
        neighbors_a = list(graph_copy.neighbors(a))
        neighbors_b = list(graph_copy.neighbors(b))
        for v in neighbors_a:
            graph_copy.add_edge(u, v, weight=graph_copy[a][v]["weight"])
        for v in neighbors_b:
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]["weight"] += graph_copy[b][v]["weight"]
            else:
                graph_copy.add_edge(u, v, weight=graph_copy[b][v]["weight"])
        graph_copy.remove_node(a)
        graph_copy.remove_node(b)

        u += 1

    # Compute the information loss of each possible merge
    u = 2 * n_nodes - 2
    merging_priority = heapdict()
    for t in list(reversed(range(n_nodes - 1))):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]

        current_tree = cluster_trees[u]
        d_ab = current_tree.d_ab
        p_ab = current_tree.sum_p_ab
        pi_a_pi_b = current_tree.sum_pi_a_pi_b

        # Loss computation with left level
        if left_tree.d_ab is not None:
            left_d_ab = left_tree.d_ab
            left_p_ab = left_tree.sum_p_ab
            left_pi_a_pi_b = left_tree.sum_pi_a_pi_b

            left_tree.up_merge_d_ab = (p_ab + left_p_ab) * (pi_a_pi_b + left_pi_a_pi_b)
            left_tree.up_merge_loss = left_tree.up_merge_d_ab - (d_ab + left_d_ab)

            merging_priority[left_tree.cluster_label] = left_tree.up_merge_loss

        # Loss computation with right level
        if right_tree.d_ab is not None:
            right_d_ab = right_tree.d_ab
            right_p_ab = right_tree.sum_p_ab
            right_pi_a_pi_b = right_tree.sum_pi_a_pi_b

            right_tree.up_merge_d_ab = (p_ab + right_p_ab) * (
                pi_a_pi_b + right_pi_a_pi_b
            )
            right_tree.up_merge_loss = right_tree.up_merge_d_ab - (d_ab + right_d_ab)

            merging_priority[right_tree.cluster_label] = right_tree.up_merge_loss

        u -= 1

    # Merge n_levels times
    while len(merging_priority) >= n_internal:
        cluster_label, minimum_loss = merging_priority.popitem()

        merged_tree = cluster_trees[cluster_label]
        father_merged_tree = merged_tree.father

        # Merge the two levels
        father_merged_tree.sum_pi_a_pi_b += merged_tree.sum_pi_a_pi_b
        father_merged_tree.sum_p_ab += merged_tree.sum_p_ab
        father_merged_tree.d_ab = merged_tree.up_merge_d_ab
        father_merged_tree.children |= merged_tree.children
        father_merged_tree.children.remove(merged_tree)
        cluster_trees.pop(cluster_label)

        # Updates the father and the children loss
        if father_merged_tree.father is not None:
            pi_a_pi_b = father_merged_tree.sum_pi_a_pi_b
            p_ab = father_merged_tree.sum_p_ab
            d_ab = father_merged_tree.d_ab
            father_pi_a_pi_b = father_merged_tree.father.sum_pi_a_pi_b
            father_p_ab = father_merged_tree.father.sum_p_ab
            father_d_ab = father_merged_tree.father.d_ab

            father_merged_tree.up_merge_d_ab = (p_ab + father_p_ab) * (
                pi_a_pi_b + father_pi_a_pi_b
            )
            father_merged_tree.up_merge_loss = father_merged_tree.up_merge_d_ab - (
                d_ab + father_d_ab
            )

            merging_priority[
                father_merged_tree.cluster_label
            ] = father_merged_tree.up_merge_loss

        for child in father_merged_tree.children:
            if child.d_ab is not None:
                pi_a_pi_b = father_merged_tree.sum_pi_a_pi_b
                p_ab = father_merged_tree.sum_p_ab
                d_ab = father_merged_tree.d_ab
                child_pi_a_pi_b = child.sum_pi_a_pi_b
                child_p_ab = child.sum_p_ab
                child_d_ab = child.d_ab

                child.up_merge_d_ab = (p_ab + child_p_ab) * (
                    pi_a_pi_b + child_pi_a_pi_b
                )
                child.up_merge_loss = child.up_merge_d_ab - (d_ab + child_d_ab)

                child.father = father_merged_tree

                merging_priority[child.cluster_label] = child.up_merge_loss

    keys = list(cluster_trees.keys())
    for t in keys:
        if len(cluster_trees[t].children) == 0:
            cluster_trees.pop(t)
    compressed_tree = {
        c.cluster_label: [child.cluster_label for child in c.children]
        for i, c in cluster_trees.items()
    }
    return compressed_tree


def networkx_from_torch_sparse(adjacency: torch.sparse_coo_tensor):
    values = adjacency.values()
    rows = adjacency.indices()[0]
    cols = adjacency.indices()[1]
    A = sp.coo_matrix((values, (rows, cols)), shape=adjacency.shape)
    return nx.from_scipy_sparse_matrix(A)


def dendrogram_to_dag(den):
    tree = nx.DiGraph()
    for i, row in enumerate(den):
        tree.add_edge(den.shape[0] + i + 1, int(row[0]))
        tree.add_edge(den.shape[0] + i + 1, int(row[1]))
    return tree


def A_B_to_dag(A, B):
    cut = B.sum(1).argmin()
    if cut == 0 or cut == B.shape[0] - 1:
        T = best_tree(A.numpy(), B.numpy())
        A, B = tree_to_A_B(T, A.shape[0], A.shape[1])
        cut = B.sum(1).argmin()
    B = B[: cut + 1, : cut + 1]
    A = A[:, : cut + 1]
    adjacency = np.zeros((A.shape[0] + A.shape[1], A.shape[0] + A.shape[1]))
    adjacency[: A.shape[0], A.shape[0] :] = A
    adjacency[A.shape[0] :, A.shape[0] :] = B
    adjacency[-1] = 0
    return nx.from_numpy_array(adjacency.T, create_using=nx.DiGraph)


def den_to_A_B(den):
    adjacency = np.zeros((den.shape[0] * 2 + 1, den.shape[0] * 2 + 1))
    for i, row in enumerate(den):
        adjacency[int(row[0]), den.shape[0] + i + 1] = 1
        adjacency[int(row[1]), den.shape[0] + i + 1] = 1
    A = adjacency[: den.shape[0] + 1, den.shape[0] + 1 :]
    B = adjacency[den.shape[0] + 1 :, den.shape[0] + 1 :]
    return A, B


# Converts a linkage matrix (i.e. dendrogram) in a tree structure.
def dendrogram_to_tree(dendrogram, n_leaves):
    tree, d = {}, {u: float("inf") for u in range(n_leaves)}
    for t in range(n_leaves - 1):
        u = dendrogram[t, 0]
        v = dendrogram[t, 1]
        dist = dendrogram[t, 2]
        if dist == d[u] and dist == d[v]:
            tree[n_leaves + t] = tree.pop(u)
            tree[n_leaves + t] += tree.pop(v)
        elif dist == d[u]:
            tree[n_leaves + t] = [v]
            tree[n_leaves + t] += tree.pop(u)
        elif dist == d[v]:
            tree[n_leaves + t] = [u]
            tree[n_leaves + t] += tree.pop(v)
        else:
            tree[n_leaves + t] = [u, v]
        d[n_leaves + t] = dist
    return tree


# Select a tree from the learned parent probabilities by choosing for each node its most likely parent.
def best_tree(A, B):
    n_leaves, n_internal = A.shape
    parents = {}

    tree = {k: [] for k in range(n_leaves, n_leaves + n_internal)}

    # Sample tree.
    for leaf in range(A.shape[0]):
        parent = np.argmax(A[leaf, :]) + A.shape[0]
        tree[parent].append(leaf)
        parents[leaf] = parent

    for internal in range(B.shape[0] - 1):
        parent = np.argmax(B[internal, :]) + A.shape[0]
        tree[parent].append(internal + A.shape[0])
        parents[internal + A.shape[0]] = parent
    # return tree
    # Prune tree.
    updated = True
    while updated:
        updated = False
        for k, v in tree.items():
            # Do not prune out the root node.
            if k == n_leaves + n_internal - 1:
                continue

            # Prune out node if it has none or only one child.
            if len(v) <= 1:
                assert k in parents
                parent = parents[k]
                if len(v) == 1:
                    # Add child to parent.
                    tree[parent].append(v[0])
                    parents[v[0]] = parent

                # Remove node from children of parent.
                tree[parent] = [n for n in tree[parent] if n != k]
                # Remove node from tree.
                del parents[k]
                del tree[k]
                updated = True
                break

    # Special handling of the root node.
    if len(tree[n_leaves + n_internal - 1]) == 1:
        child = tree[n_leaves + n_internal - 1][0]
        tree[n_leaves + n_internal - 1] = tree[child]
        del tree[child]
        del parents[child]
        for child in tree[n_leaves + n_internal - 1]:
            parents[child] = n_leaves + n_internal - 1
    return tree


def tree_to_dendrogram(tree, n_leaves, height_map=None):
    tree = [(k, list(v)) for k, v in tree.items()]

    dendrogram = []
    dendro_map = {i: i for i in range(n_leaves)}
    link_map = {}  # Needed for plotting.

    mod = True

    while len(tree) > 0 and mod:
        mod = False
        for idx, (k, v) in enumerate(tree):
            if height_map != None:
                height = height_map[k - n_leaves] + 1
            else:
                height = k - n_leaves + 1

            # Check if we can add node to dendrogram.
            if not all(n in dendro_map for n in v):
                continue

            mod = True
            # This creates the parent node.
            link_map[len(dendrogram)] = k - n_leaves

            dendrogram.append((dendro_map[v[0]], dendro_map[v[1]], height, height))

            for n in v[2::]:
                n1 = n_leaves - 1 + len(dendrogram)
                n2 = dendro_map[n]
                link_map[len(dendrogram)] = k - n_leaves
                dendrogram.append((n1, n2, height, height))

            dendro_map[k] = n_leaves - 1 + len(dendrogram)
            tree.pop(idx)
            break

    if len(tree) > 0:
        raise Exception("Invalid tree!")

    return np.array(dendrogram, float), link_map


def chunker(seq, size):
    """
    Chunk a list into chunks of size `size`.
    From
    https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks

    Parameters
    ----------
    seq: input list
    size: size of chunks

    Returns
    -------
    The list of lists of size `size`
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def get_initial_hierarchy(data_module, internal_nodes, loss, **kwargs):
    adjacency = data_module.dataset.adjacency
    graph = networkx_from_torch_sparse(adjacency)
    den = agglomerative_linkage(
        graph, affinity="unitary", linkage="average", check=True
    )
    compressed = compress_hierarchy_dasgupta(graph, den, internal_nodes)
    return tree_to_A_B(compressed, adjacency.shape[0], internal_nodes)


def compute_skn_tsd(graph, T):
    adj = graph.adj
    A_sp = sp.csr_matrix((adj.values().cpu(), adj.indices().cpu()), shape=adj.shape)
    den, _ = tree_to_dendrogram(T, adj.shape[0])
    skn_TSD_raw = tree_sampling_divergence(A_sp, den, normalized=False)
    return (100 * skn_TSD_raw / graph.mutual_information).item()
