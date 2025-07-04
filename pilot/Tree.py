from __future__ import annotations

import uuid
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import textwrap 

from networkx.drawing.nx_pydot import graphviz_layout

from collections import Counter 


class tree(object):
    """
    We use a tree object to save the PILOT model.
    Attributes:
    -----------
    node: str,
        type of the regression model
        'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
    pivot: tuple,
        a tuple to indicate where we performed a split. The first
        coordinate is the feature_id and the second one is
        the pivot.
    lm_l: ndarray,
        a 1D array to indicate the linear model for the left child node. The first element
        is the coef and the second element is the intercept.
    lm_r: ndarray,
        a 1D array to indicate the linear model for the right child node. The first element
        is the coef and the second element is the intercept.
    Rt: float,
        a real number indicating the rss in the present node.
    depth: int,
        the depth of the current node/subtree
    interval: ndarray,
        1D float array for the range of the selected predictor in the training data
    pivot_c: ndarry,
        1D int array. Indicating the levels in the left node
        if the selected predictor is categorical
    """

    def __init__(
        self,
        node=None,
        pivot=None,
        lm_l=None,
        lm_r=None,
        Rt=None,
        depth=None,
        interval=None,
        pivot_c=None,
    ) -> None:
        """
        Here we input the tree attributes.
        parameters:
        ----------
        node: str,
            type of the regression model
            'lin', 'blin', 'pcon', 'plin' or 'END' to denote the end of the tree
        pivot: tuple,
            a tuple to indicate where we performed a split. The first
            coordinate is the feature_id and the second one is
            the pivot.
        lm_l: ndarray,
            a 1D array to indicate the linear model for the left child node. The first element
            is the coef and the second element is the intercept.
        lm_r: ndarray,
            a 1D array to indicate the linear model for the right child node. The first element
            is the coef and the second element is the intercept.
        Rt: float,
            a real number indicating the rss in the present node.
        depth: int,
            the depth of the current node/subtree
        interval: ndarray,
            1D float array for the range of the selected predictor in the training data
        pivot_c: ndarry,
            1D int array. Indicating the levels in the left node
            if the selected predictor is categorical
        """
        self.left = None  # go left by default if node is 'lin'
        self.right = None
        self.Rt = Rt
        self.node = node
        self.pivot = pivot
        self.lm_l = lm_l
        self.lm_r = lm_r
        self.depth = depth
        self.interval = interval
        self.pivot_c = pivot_c

        # Define which nodes are split nodes
        SPLIT_NODE_TYPES = ["pcon", "plin", "blin", "pconc", "lasso_split"]

    def nodes_selected(self, depth=None) -> dict[str, int]:
        """Recursively counts the number of each model type selected in the tree.

        This method traverses the tree from the current node downwards and aggregates
        the counts of all model types encountered (e.g., 'pcon', 'lin').

        Args:
            depth (int, optional): If specified, the count will only include nodes
                up to and including this depth level. Defaults to None, which counts
                all nodes in the subtree.

        Returns:
            dict[str, int]: A dictionary (or collections.Counter) mapping model type
                names to their frequency in the tree.
        """
        # 1. Handle base cases for recursion. Return an empty Counter.
        if self.node == "END":
            return Counter()

        # This stops the count *before* processing a node at the forbidden depth.
        if depth is not None and self.depth is not None and self.depth >= depth + 1:
            return Counter()

        # 2. Get the counts from the left subtree.
        #    If no left child, start with an empty Counter.
        total_counts = self.left.nodes_selected(depth) if self.left is not None else Counter()

        # 3. Add the current node to the counts.
        total_counts[self.node] += 1

        # 4. If it's a split node, get counts from the right and add them.
        if self.node in self.SPLIT_NODE_TYPES and self.right is not None:
            right_counts = self.right.nodes_selected(depth)
            total_counts.update(right_counts)  # .update() is the correct way to add Counters

        return total_counts

    @staticmethod
    def get_depth(model_tree):
        """Recursively calculates the maximum depth of a PILOT tree.

        The depth is defined by the longest path from the given `model_tree`
        node to a terminal 'END' node.

        Args:
            model_tree (tree): The root node of the tree or subtree for which
                to calculate the depth.

        Returns:
            int: The maximum depth of the tree.
        """
        depth = model_tree.depth
        left = model_tree.left
        right = model_tree.right
        if left is not None and left.node != "END":
            depth = tree.get_depth(left)
        if right is not None and right.node != "END":
            depth = max(depth, tree.get_depth(right))
        return depth
        
class tree_multi(tree):
    """An extended tree node that supports multivariate linear models.

    This class inherits all properties from the base `tree` class but adds
    attributes to store more complex, multivariate models (e.g., from LASSO)
    for both the left and right children of a split. This allows the tree to
    represent not just a split condition, but also the rich local models
    that apply to each resulting partition.

    Attributes:
        multi_model_indices_L (np.ndarray): Array of feature indices used in the
            multivariate model for the left child branch.
        multi_model_coeffs_L (np.ndarray): Array of coefficients corresponding to the
            indices for the left child's model.
        multi_model_intercept_L (float): The intercept term for the left child's model.
        multi_model_indices_R (np.ndarray): Array of feature indices used in the
            multivariate model for the right child branch.
        multi_model_coeffs_R (np.ndarray): Array of coefficients corresponding to the
            indices for the right child's model.
        multi_model_intercept_R (float): The intercept term for the right child's model.
        best_node_prev (str): Stores the original, simpler node type (e.g., 'pcon')
            that was upgraded to a multivariate node, preserving model history.
    """
    def __init__(self, multi_model_indices_L=None, multi_model_coeffs_L=None, multi_model_intercept_L=None, 
                 multi_model_indices_R=None, multi_model_coeffs_R=None, multi_model_intercept_R=None, 
                 best_node_prev=None, **kwargs):
        """Initializes a multivariate-supporting tree node.

       This node extends the base `tree` class by adding attributes to store
       multivariate linear models for its children, which are typically found
       using methods like LASSO.

       Args:
           multi_model_indices_L (np.ndarray, optional): Array of feature indices for the
               left child's multivariate model. Defaults to None.
           multi_model_coeffs_L (np.ndarray, optional): Array of coefficients for the
               left child's multivariate model. Defaults to None.
           multi_model_intercept_L (float, optional): Intercept for the left child's
               multivariate model. Defaults to None.
           multi_model_indices_R (np.ndarray, optional): Array of feature indices for the
               right child's multivariate model. Defaults to None.
           multi_model_coeffs_R (np.ndarray, optional): Array of coefficients for the
               right child's multivariate model. Defaults to None.
           multi_model_intercept_R (float, optional): Intercept for the right child's
               multivariate model. Defaults to None.
           best_node_prev (str, optional): The original node type before it was potentially
               upgraded to a multivariate split (e.g., 'pconc'). Defaults to None.
           **kwargs: Keyword arguments passed to the parent `tree` class initializer.
        """
        
        # Call the parent class's __init__ to set up all the basic attributes
        super().__init__(**kwargs)

        # Add the new, specific attributes for multivariate models
        self.multi_model_indices_L = multi_model_indices_L
        self.multi_model_coeffs_L = multi_model_coeffs_L
        self.multi_model_intercept_L = multi_model_intercept_L
        self.multi_model_indices_R = multi_model_indices_R
        self.multi_model_coeffs_R = multi_model_coeffs_R
        self.multi_model_intercept_R = multi_model_intercept_R
        self.best_node_prev = best_node_prev


def _get_child_data(training_data, model_tree, feature_names=None):
    """Splits training data into left and right subsets based on a node's rule.

    This helper function reads the split information (pivot value, pivot
    categories) from a `model_tree` node and partitions the `training_data`
    accordingly. It handles both numerical and categorical splits. If the node
    is not a split node, all data is assigned to the left child.

    Args:
        training_data (np.ndarray): The data samples at the current node.
        model_tree (tree or tree_multi): The node containing the split rule.
        feature_names (list[str], optional): A list of feature names. Not currently
            used in this function but kept for API consistency. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray or None]: A tuple containing the data for
            the left child and the right child. `right_data` is `None` if the node
            is not a split node.
    """
    left_data = training_data
    right_data = None # Will be populated for split nodes

    # If node is in no split all training data code to the left data
    if model_tree.node in ["END", "con", "lin", "lasso_no_split"]:
        return left_data, right_data

    # Determine the split feature index
    split_feature_idx = model_tree.pivot[0]

    # Handle categorical splits (pconc or lasso_split upgraded from pconc)
    is_categorical_split = model_tree.node == "pconc" or \
                           (model_tree.node == "lasso_split" and
                            hasattr(model_tree, 'best_node_prev') and
                            model_tree.best_node_prev == 'pconc')

    if is_categorical_split:
        left_mask = np.isin(training_data[:, split_feature_idx], model_tree.pivot_c)
        left_data = training_data[left_mask]
        right_data = training_data[~left_mask] 
    # Handle numerical splits
    else:
        split_value = model_tree.pivot[1]
        left_mask = training_data[:, split_feature_idx] <= split_value
        left_data = training_data[left_mask]
        right_data = training_data[~left_mask]

    return left_data, right_data

def _format_model_equation(intercept, coeffs, indices, pivot_feat_idx=None, max_feats=3, add_plus_sign=False):
    """Formats a linear model's parameters into a readable string equation.

    This utility function creates a string like "+ 3.14 + 1.50*X1 - 2.00*X5"
    from model coefficients and indices. It can format constant, univariate,
    and multivariate models.

    Args:
        intercept (float): The model's intercept term.
        coeffs (np.ndarray or float): The model's coefficient(s).
        indices (np.ndarray, optional): For multivariate models, the indices of the
            features corresponding to the coefficients. Defaults to None.
        pivot_feat_idx (int, optional): For univariate models, the index of the
            single feature. Defaults to None.
        max_feats (int, optional): The maximum number of feature terms to show in a
            multivariate equation before adding "...". Defaults to 3.
        add_plus_sign (bool, optional): If True, a '+' sign is added for positive
            intercepts to show continuity in an equation. Defaults to False.

    Returns:
        str: The formatted model equation as a string.
    """
    if intercept is None:
        return ""

    sign = ""
    if add_plus_sign and intercept >= 0:
        sign = "+ "
    elif intercept < 0:
        sign = "- "
    
    equation = f"{sign}{abs(intercept):.3f}"

    # Handle multivariate models
    if coeffs is not None and indices is not None and indices.size > 0:
        active_coeffs = []
        for i, idx in enumerate(indices):
            feat_name = f"X{int(idx) + 1}"
            active_coeffs.append((coeffs[i], feat_name))
        
        for i, (coeff, name) in enumerate(active_coeffs[:max_feats]):
            term_sign = "+ " if coeff >= 0 else "- "
            equation += f" {term_sign}{abs(coeff):.2f}*{name}"
        
        if len(active_coeffs) > max_feats:
            equation += " ..."
    
    # Handle univariate linear models (where pivot_feat_idx is provided)
    elif pivot_feat_idx is not None and coeffs is not None and abs(coeffs) > 1e-6:
        term_sign = "+ " if coeffs >= 0 else "- "
        name = f"X{int(pivot_feat_idx) + 1}"
        equation += f" {term_sign}{abs(coeffs):.3f}*{name}"
            
    return equation

def _construct_graph_recursive(model_tree, G, training_data, feature_names, parent_id=None):
    """Recursively traverses a PILOT tree to build a NetworkX graph.

    This function walks through the `model_tree`, creating a node in the
    NetworkX graph `G` for each node in the PILOT tree. It stores all model
    attributes (type, pivot, coefficients, etc.) in the graph nodes and
    connects them with edges to represent the tree structure.

    Args:
        model_tree (tree or tree_multi): The current node in the PILOT tree.
        G (nx.DiGraph): The NetworkX graph being built (modified in-place).
        training_data (np.ndarray): The subset of data corresponding to the current node.
        feature_names (list[str]): A list of feature names.
        parent_id (str, optional): The unique ID of the parent node in the graph `G`.
            Defaults to None (for the root).
    """
    if model_tree is None or model_tree.node is None:
        return

    current_id = str(uuid.uuid4().fields[-1])[-6:]
    node_name = f"{model_tree.node}_{current_id}"

    # Add node to the graph with possible multivariate attributes
    G.add_node(
        node_name,
        node_type=model_tree.node,
        depth=model_tree.depth,
        samples=len(training_data) if training_data is not None else 0,
        pivot=model_tree.pivot,
        pivot_c=model_tree.pivot_c,
        lm_l=model_tree.lm_l,
        lm_r=model_tree.lm_r,
        multi_indices_L=getattr(model_tree, 'multi_model_indices_L', None),
        multi_coeffs_L=getattr(model_tree, 'multi_model_coeffs_L', None),
        multi_intercept_L=getattr(model_tree, 'multi_model_intercept_L', None),
        multi_indices_R=getattr(model_tree, 'multi_model_indices_R', None),
        multi_coeffs_R=getattr(model_tree, 'multi_model_coeffs_R', None),
        multi_intercept_R=getattr(model_tree, 'multi_model_intercept_R', None),
        best_node_prev=getattr(model_tree, 'best_node_prev', None)
    )

    # Add edge to parent node
    if parent_id is not None:
        G.add_edge(parent_id, node_name)

    # Recursive calls for that branch of the tree are finished when leaf node is reached
    if model_tree.node == "END":
        return

    # Make recursive calls
    left_data, right_data = _get_child_data(training_data, model_tree, feature_names)
    if model_tree.left:
        _construct_graph_recursive(model_tree.left, G, left_data, feature_names, parent_id=node_name)
    if model_tree.right:
        _construct_graph_recursive(model_tree.right, G, right_data, feature_names, parent_id=node_name)

def get_node_depths(G, root_node):
    """Calculates the depth of each node in a graph starting from a root.

    This function performs a Breadth-First Search (BFS) to determine the
    shortest path distance (depth) from the `root_node` to all other reachable
    nodes in the graph `G`.

    Args:
        G (nx.DiGraph): The graph to traverse.
        root_node: The starting node for the BFS traversal.

    Returns:
        dict: A dictionary mapping each node's ID to its integer depth.
    """
    depths = {root_node: 0}
    queue = [(root_node, 0)]
    while queue:
        u, d = queue.pop(0)
        for v in G.successors(u):
            if v not in depths:
                depths[v] = d + 1
                queue.append((v, d + 1))
    return depths

def visualize_pilot_tree(model_tree, training_data, feature_names: list[str] = None,
                         figsize: tuple = (40, 25), filename: str = None,
                         pre_model=None, pre_model_name: str = None,
                         pre_model_feature_names: list[str] = None,
                         categorical_feature_names: list[str] = None):
    """Creates and displays a detailed, publication-quality visualization of a PILOT model tree.

    This function translates a trained `model_tree` object into a rich visual
    representation using NetworkX and Matplotlib. It distinguishes between
    split nodes and leaf nodes, labels edges with model equations, and provides
    a legend for feature interpretability. The visualization correctly handles
    simple and multivariate models, as well as numerical and categorical splits.

    Args:
        model_tree (tree or tree_multi): The trained PILOT tree object to visualize.
        training_data (np.ndarray): The full training dataset used to build the tree.
        feature_names (list[str], optional): List of names for the predictors, used
            for the legend. Defaults to None.
        figsize (tuple, optional): The size of the figure for the plot.
            Defaults to (40, 25).
        filename (str, optional): If provided, the plot will be saved to this path.
            Defaults to None.
        pre_model (object, optional): A trained scikit-learn-like model (e.g., a global
            ensemble) to display at the root. Must have `coef_` and `intercept_`
            attributes. Defaults to None.
        pre_model_name (str, optional): The name of the `pre_model` to display.
            Defaults to None.
        pre_model_feature_names (list[str], optional): The full list of feature names
            (including one-hot encoded ones) used by the `pre_model`. Defaults to None.
        categorical_feature_names (list[str], optional): List of base names for
            categorical features, used to correctly parse the `pre_model_feature_names`.
            Defaults to None.
    """
    if not model_tree or model_tree.node == "END":
        print("Tree is empty or just a single constant model. Nothing to plot.")
        return

    # 1. Build the full graph with all intermediate nodes
    G = nx.DiGraph()
    G.add_node("Root", node_type="Root", samples=len(training_data))
    _construct_graph_recursive(model_tree, G, training_data, feature_names, parent_id="Root")

    # 2. Prepare for Drawing
    plt.figure(figsize=figsize); ax = plt.gca()
    pos = graphviz_layout(G, prog="dot")
    
    # 3. Node Classification BASED ON STRUCTURE (out_degree)
    split_nodes = [n for n in G.nodes() if G.out_degree(n) > 1 and n != "Root"]
    leaf_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'END']
    visible_nodes = set(split_nodes + leaf_nodes + ["Root"])
    passthrough_nodes = [n for n in G.nodes() if n not in visible_nodes]

    number_of_visible_nodes = len(visible_nodes)

    # Make the text spacing adjust to number of nodes
    if number_of_visible_nodes < 8:
        adjustment_text = 5
    elif number_of_visible_nodes < 25:
        adjustment_text = 8
    else:
        adjustment_text = 12
    
    # Node Styling
    NODE_SIZE = 9000
    nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_size=NODE_SIZE, node_shape='o', node_color='lightgreen', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=["Root"], node_size=NODE_SIZE, node_shape='s', node_color='lightgray', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=passthrough_nodes, node_size=1, node_color='none')

    # 4. Create and Draw Node Labels for VISIBLE nodes
    node_labels = {}
    for n in visible_nodes:
        data = G.nodes[n]
        samples = data.get('samples')

        # Make root node
        if n == "Root":
            if pre_model and pre_model_name and hasattr(pre_model, 'coef_'):
                # If a pre-model is provided, format its detailed equation.
                intercept = pre_model.intercept_
                coeffs = pre_model.coef_.flatten()
                name_to_index_map = {name: i for i, name in enumerate(feature_names)}

                # Create a list of (importance, display_name, signed_coefficient)
                # to sort by importance for feature legend.
                feature_effects = []
                for i, full_name in enumerate(pre_model_feature_names):
                    signed_coef = coeffs[i]
                    if abs(signed_coef) > 1e-6:
                        display_name = full_name
                        is_categorical_part = False

                        # Check if it's a one-hot encoded feature
                        for cat_name in categorical_feature_names:
                            if full_name.startswith(cat_name + '_'):
                                original_idx = name_to_index_map[cat_name]
                                class_name = full_name[len(cat_name) + 1:]
                                # Format as X2_(red)
                                display_name = f"X{original_idx + 1}_({class_name})"
                                is_categorical_part = True
                                break

                        # If it was not a categorical part, it's a numerical feature
                        if not is_categorical_part:
                            original_idx = name_to_index_map[full_name]
                            # Format as X5
                            display_name = f"X{original_idx + 1}"

                        feature_effects.append((abs(signed_coef), display_name, signed_coef))

                # Sort features by absolute coefficient value, descending
                feature_effects.sort(key=lambda x: x[0], reverse=True)

                # Build the equation string with a max of 5 features
                equation = f"{pre_model_name} Model\n"
                equation += f"{intercept:.2f}"

                count = 0
                for abs_coef, display_name, signed_coef in feature_effects:
                    if count < 5:
                        sign = "+" if signed_coef >= 0 else "-"
                        equation += f" {sign} {abs_coef:.2f}*{display_name}"
                    else:
                        equation += "+ ..."
                        break
                    count += 1

                equation += f"\n({samples} samples)"
                node_labels[n] = equation
            else:
                # Default behavior if no pre-model
                node_labels[n] = f"Root\n({samples} samples)"
        # Make leaf nodes
        elif data.get('node_type') == 'END':
            node_labels[n] = f"Leaf\n({samples} samples)"
        # Make split nodes
        elif n in split_nodes:
            pivot = data.get('pivot')
            pivot_c = data.get('pivot_c') # This will be an array or None
            
            current_node = data.get('node_type')
            prev_node = ""
            if hasattr(data, 'get') and data.get('best_node_prev') is not None:
                prev_node = data.get('best_node_prev')
            
            if pivot:  # Check if a split feature exists
                feature_index = pivot[0]
                feature_name = f"X{feature_index + 1}"
            
                # Now, check if it's categorical by looking at pivot_c
                if pivot_c is not None and (current_node == 'pconc' or (current_node == 'lasso_split' and prev_node == 'pconc')):
                    # It's a CATEGORICAL split!
                    # Format the levels nicely for display
                    levels_to_show = list(pivot_c)
                    if len(levels_to_show) > 3:
                        label_text = f"{feature_name} in [{levels_to_show[0]}, ..., {levels_to_show[-1]}]"
                    else:
                        label_text = f"{feature_name} in {levels_to_show}"
                else:
                    # It's a NUMERICAL split
                    split_value = pivot[1]
                    label_text = f"{feature_name} <= {split_value:.2f}"
            
                # Set the final node label
                node_labels[n] = f"{label_text}\n({data.get('samples')} samples)"

    split_node_labels = {n: lbl for n, lbl in node_labels.items() if n in split_nodes}
    leaf_and_root_labels = {n: lbl for n, lbl in node_labels.items() if n not in split_nodes}

    # Make a box for split nodes for better styling
    split_node_bbox = dict(boxstyle="round,pad=0.5", fc="skyblue", ec="black")

    # Draw the split node labels WITH the bbox, which creates the rectangular shape
    nx.draw_networkx_labels(G, pos,
                            labels=split_node_labels,
                            font_size=12,
                            font_weight='bold',
                            bbox=split_node_bbox)

    # Draw the other labels (leaves and root) without a special bbox
    nx.draw_networkx_labels(G, pos,
                            labels=leaf_and_root_labels,
                            font_size=12,
                            font_weight='bold')
  
    # 5. Trace "Virtual" Edges and Draw Single Arrows + All Text
    for u in ["Root"] + split_nodes:
        successors = list(G.successors(u))
        
        # Determine the name of the visually leftmost child node. This is reliable.
        leftmost_child_name = None
        if len(successors) == 1:
            # If there's only one child, by convention, it's a "left" branch (for no-split models).
            leftmost_child_name = successors[0]
        elif len(successors) == 2:
            # For a true split, compare the x-positions of the two siblings.
            child1, child2 = successors[0], successors[1]
            if pos[child1][0] < pos[child2][0]:
                leftmost_child_name = child1
            else:
                leftmost_child_name = child2
                
        for v_start in successors:
            final_dest = v_start
            model_chain = []
            
            # Trace the path until we hit the next visible node
            path_tracer = v_start
            while path_tracer not in visible_nodes:
                model_chain.append(G.nodes[path_tracer])
                successors = list(G.successors(path_tracer))
                if not successors: break
                path_tracer = successors[0]
            final_dest = path_tracer
            
            # Draw one arrow for the entire path segment
            nx.draw_networkx_edges(G, pos, edgelist=[(u, final_dest)], arrowsize=30, node_size=NODE_SIZE, width=1.5, edge_color='darkgray')
            
            # --- Collect and draw all text for this path segment ---
            x1, y1 = pos[u]; x2, y2 = pos[final_dest]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            y_offset = 0

            bbox_props = dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.3, edgecolor='none')

            # Get model from parent 'u' if it's a split node
            if u in split_nodes:
                parent_data = G.nodes[u]
                is_left = (v_start == leftmost_child_name)
                yes_no = "Yes" if is_left else "No"
                
                # Get the correct model (L/R) from the split node
                lm, inter, coeffs, inds = (parent_data.get('lm_l'), parent_data.get('multi_intercept_L'), parent_data.get('multi_coeffs_L'), parent_data.get('multi_indices_L')) if is_left else (parent_data.get('lm_r'),parent_data.get('multi_intercept_R'), parent_data.get('multi_coeffs_R'), parent_data.get('multi_indices_R'))
                
                # Make nice equations for split functions
                model_text = ""
                if lm is not None and parent_data.get('node_type') in ["pcon", "plin", "blin", "pconc"]: model_text = _format_model_equation(lm[1], lm[0], None, pivot_feat_idx=parent_data.get('pivot')[0], add_plus_sign=True)
                elif inter is not None: model_text = _format_model_equation(inter, coeffs, inds, add_plus_sign=True)

                if is_left:
                    mid_y += adjustment_text - 5

                # Display the split function
                display_text = f"{yes_no}  {model_text}"
                ax.text(mid_x, mid_y - y_offset, display_text, ha='center', va='center', color='red', fontsize=12,
                        bbox=bbox_props)
                y_offset += adjustment_text
            
            # Get models from the intermediate chain
            for model_data in model_chain:
                node_type = model_data.get('node_type')
                # Pass-through nodes only have a left model
                lm, inter, coeffs, inds = model_data.get('lm_l'), model_data.get('multi_intercept_L'), model_data.get('multi_coeffs_L'), model_data.get('multi_indices_L')

                # Make nice equations for no-split functions
                model_text = ""
                if lm is not None and node_type in ["con", "lin"]: model_text = _format_model_equation(lm[1], lm[0], None, pivot_feat_idx=model_data.get('pivot')[0] if model_data.get('node_type') != 'con' else None,                    add_plus_sign=True)
                elif inter is not None: model_text = _format_model_equation(inter, coeffs, inds, add_plus_sign=True)
                
                ax.text(mid_x, mid_y - y_offset, model_text, ha='center', va='center', color='red', fontsize=12, bbox=bbox_props)
                y_offset += adjustment_text

    # 6. Final Touches
    plt.title("PILOT Model Tree Visualization", size=30); plt.tight_layout()
    # Add legend
    if feature_names:
        final_legend_indices = set()

        # Step A: Get all features used in the PILOT TREE
        def find_all_indices(node, index_set):
            if node is None or node.node == "END": return
            is_active_model = False

            # Check if model is actually used in final tree
            if node.node not in ["con", "lin", "lasso_no_split", "END"]:
                is_active_model = True
            elif node.node in ["lin", "lasso_no_split"]:
                if node.left and node.left.node != "END":
                    is_active_model = True

            # Get the features used in the model
            if is_active_model:
                if node.pivot and node.pivot[0] is not None:
                    index_set.add(int(node.pivot[0]))
                if hasattr(node, 'multi_model_indices_L') and node.multi_model_indices_L.size > 0:
                    index_set.update(node.multi_model_indices_L.astype(int))
                if hasattr(node, 'multi_model_indices_R') and node.multi_model_indices_R.size > 0:
                    index_set.update(node.multi_model_indices_R.astype(int))

            # Recursive calls
            if node.left: find_all_indices(node.left, index_set)
            if node.right: find_all_indices(node.right, index_set)

        find_all_indices(model_tree, final_legend_indices)

        # Step B: Get the TOP 5 features from the PRE-MODEL
        if pre_model and hasattr(pre_model, 'coef_'):
            coeffs = np.abs(pre_model.coef_.flatten())
            name_to_index_map = {name: i for i, name in enumerate(feature_names)}

            agg_imp = {name: 0.0 for name in feature_names}
            for i, full_name in enumerate(pre_model_feature_names):
                original_name = full_name
                for cat_name in categorical_feature_names:
                    if full_name.startswith(cat_name + '_'):
                        original_name = cat_name
                        break
                agg_imp[original_name] += coeffs[i]

            sorted_features = sorted(agg_imp.items(), key=lambda item: item[1], reverse=True)

            N = 5  # Set to Top 5
            for name, importance_score in sorted_features[:N]:
                if importance_score > 1e-6:
                    final_legend_indices.add(name_to_index_map[name])

        # Step C: Build the legend text from the combined set
        legend_title = "Feature Legend"
        if pre_model:
            legend_title += " (Top Ensemble Features)"
        legend_text = f"{legend_title}:\n" + "-" * 45 + "\n"

        for idx in sorted(list(final_legend_indices)):
            if 0 <= idx < len(feature_names):
                legend_text += f"X{idx + 1}: {feature_names[idx]}\n"

        ax.text(0.01, 0.99, legend_text, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', color='dimgray', alpha=0.7,
                bbox=dict(boxstyle='round,pad=1.5', fc='wheat', alpha=0.4))

    # Save the plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")

    plt.close()