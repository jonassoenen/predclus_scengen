import copy
from collections import deque
from typing import Tuple, Optional, List

import numpy as np


class VisTree:
    def __init__(self, root_node):
        self.root_node = root_node

    @classmethod
    def from_sklearn_tree(cls, sklearn_tree, attributes, feature_names):
        instances = np.arange(0, attributes.shape[0], 1)
        root_node = sklearn_tree_to_nodes(sklearn_tree, 0, instances, attributes, feature_names)
        return VisTree(root_node)

    def plot_tree(self):
        pass

    def print_tree(self):
        self.root_node.print(prefix = '')

    def compress_tree(self):
        return VisTree(get_tree_with_compressed_splits(self.root_node))

    def __eq__(self, other):
        if not isinstance(other, VisTree):
            return False
        return self.root_node == other.root_node


class Node:
    def __init__(self, instances, children, split_attribute_name):
        self.children: Optional[List[Tuple[int, int, Node]]] = children
        self.instances: np.ndarray = instances
        self.split_attribute_name: Optional[str] = split_attribute_name

    @property
    def is_leaf_node(self):
        return self.children is None

    def print(self, prefix):
        if self.is_leaf_node:
            pass
        else:
            for lower, upper, child in self.children:
                if np.isinf(lower) and len(self.children) == 2:
                    test_str = f"{self.split_attribute_name} <= {upper}"
                elif np.isinf(upper) and len(self.children) == 2:
                    test_str = f"{self.split_attribute_name} > {lower}"
                else:
                    test_str = f"{lower} < {self.split_attribute_name} <= {upper}"
                print(f"{prefix} -  {test_str}")
                child.print(prefix + '\t')

    def __hash__(self):
        return hash((self.instances, self.split_attribute_name, self.children))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        elif self.is_leaf_node:
            return (
                    self.is_leaf_node == other.is_leaf_node
                    and
                    np.array_equal(self.instances, other.instances)
            )
        else:
            equals = (
                    np.array_equal(self.instances, other.instances) and
                    self.split_attribute_name == other.split_attribute_name
            )
            if not equals:
                return False
            return self.children == other.children


def get_tree_with_compressed_splits(tree_node: Node):
    if tree_node.is_leaf_node:
        return tree_node

    # check whether we can compress the current split
    # the split can be compressed if one of the children splits on the same attribute as this node
    can_compress = False
    current_attr = tree_node.split_attribute_name
    for _, _, child in tree_node.children:
        next_attr = child.split_attribute_name
        if next_attr == current_attr:
            can_compress = True
            break

    if not can_compress:
        # if you cannot compress, just retain this node and compress the subtrees
        new_children = [(lower, upper, get_tree_with_compressed_splits(child)) for lower, upper, child in
                        tree_node.children]
        return Node(tree_node.instances, new_children, current_attr)
    else:
        # if you can compress, compress the first split and compress the new subtrees further
        subtrees = compress_first_split(tree_node)
        new_children = [(lower, upper, get_tree_with_compressed_splits(subtree)) for lower, upper, subtree in
                        sorted(subtrees)]
        return Node(tree_node.instances, new_children, current_attr)


def compress_first_split(tree_to_compress: Node):
    attribute_to_compress = tree_to_compress.split_attribute_name
    subtrees = []
    queue = deque([(-np.infty, np.infty, tree_to_compress)])
    while len(queue) > 0:
        current_lower, current_upper, tree = queue.pop()
        if tree.is_leaf_node:
            subtrees.append((current_lower, current_upper, tree))
        else:
            # this is a tree
            if tree.split_attribute_name != attribute_to_compress:
                # reached a split on another attribute, stop compression here
                subtrees.append((current_lower, current_upper, copy.deepcopy(tree)))
            else:
                # this is a split on attribute to compress, so continue to compress
                for lower, upper, subtree in tree.children:
                    new_lower = max(current_lower, lower)
                    new_upper = min(current_upper, upper)
                    queue.append((new_lower, new_upper, subtree))
    return subtrees


def sklearn_tree_to_nodes(tree, node_idx, instances, attributes, feature_names):
    feature_idx = tree.feature
    feature_threshold = tree.threshold

    is_leaf_node = tree.children_left[node_idx] == tree.children_right[node_idx]
    if is_leaf_node:
        return Node(instances, None, None)
    else:
        test = attributes[:, feature_idx] <= feature_threshold
        left_attributes = attributes[test]
        left_instances = instances[test]
        left_node_idx = tree.children_left[node_idx]
        left_subtree = sklearn_tree_to_nodes(tree, left_node_idx, left_instances, left_attributes, feature_names)

        right_attributes = attributes[~test]
        right_instances = instances[~test]
        right_node_idx = tree.children_right[node_idx]
        right_subtree = sklearn_tree_to_nodes(tree, right_node_idx, right_instances, right_attributes, feature_names)

        children = [
            (None, feature_threshold, left_subtree),
            (feature_threshold, None, right_subtree)
        ]
        return Node(instances, children, feature_names[feature_idx])
