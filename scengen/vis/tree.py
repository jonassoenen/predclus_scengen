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

    def plot_tree(self, figsize = (20,25)):
        import matplotlib.pyplot as plt
        # figure configuration
        plt.figure(figsize = figsize)
        plt.axis('off')

        # find all children in order
        leafs = []
        queue = deque([self.root_node])
        while len(queue)>0:
            node = queue.pop()
            if node.is_leaf_node:
                leafs.append(node)
            else:
                queue.extend(child for _, _, child in node.children)

        x_pos = 400
        y_interval = 10
        x_interval = 30

        # first position the leaf nodes
        positions = {leaf: (x_pos, y) for leaf, y in zip(leafs, range(0, len(leafs)*y_interval + 1, y_interval))}

        # draw the leaf nodes
        for leaf, (x, y) in positions.items():
            plt.text(x, y, 'leaf', horizontalalignment='center', verticalalignment='center',
                 bbox=dict(boxstyle="Circle, pad=0.2", facecolor='white'))

        # check which parents can be drawn
        # a parent can be drawn if all of its children have a known position
        parents_to_draw = set()
        for leaf in leafs:
            if leaf.parent is None:
                continue
            siblings = leaf.parent.children
            if all(sibling in positions for _, _, sibling in siblings):
                parents_to_draw.add(leaf.parent)


        # draw parents one after another
        queue = deque(parents_to_draw)
        while len(queue)>0:
            node_to_draw = queue.popleft()

            # determine parent position based on children positions
            children_positions = [positions[child] for _, _, child in node_to_draw.children]
            split_descriptions = list(node_to_draw.split_strings)
            y_min = min(y for x, y in children_positions)
            y_max = max(y for x, y in children_positions)
            x_min = min(x for x, y in children_positions)

            # draw horizontal line from every child with splitting criterion
            for (x, y), split in zip(children_positions, split_descriptions):
                plt.plot([x, x_min - x_interval], [y, y], color='black')
                plt.text((x_min - x_interval) + x_interval / 3, y, split, verticalalignment='bottom',
                         horizontalalignment='left')

            # plot vertical line connecting all horizontal lines
            plt.plot([x_min - x_interval, x_min - x_interval], [y_min, y_max], color='black')

            # attribute text in the middle of the split
            plt.text(x_min - x_interval, (y_min + y_max) / 2, node_to_draw.split_attribute_name, verticalalignment='center',
                     horizontalalignment='center', bbox={'facecolor': 'white', 'pad': 2})

            # Save the position
            positions[node_to_draw] = (x_min - x_interval, (y_min + y_max) / 2)

            ## Check if parent becomes draweable
            if node_to_draw.parent is not None and all(child in positions for _, _, child in node_to_draw.parent.children):
                queue.append(node_to_draw.parent)


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

        # initialize parent pointers in children
        if self.children is not None:
            for _, _, child in self.children:
                child.parent = self

        self._parent: Optional[Node] = None


    def bounds_to_split_str(self, lower, upper):
        if np.isinf(lower) and len(self.children) == 2:
            return f"{self.split_attribute_name} <= {upper}"
        elif np.isinf(upper) and len(self.children) == 2:
            return f"{self.split_attribute_name} > {lower}"
        else:
            return f"{lower} < {self.split_attribute_name} <= {upper}"

    @property
    def split_strings(self):
        for lower, upper, _ in self.children:
            yield self.bounds_to_split_str(lower, upper)
    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new_parent):
        if not isinstance(new_parent, Node):
            raise ValueError

        # check if this node is child of the given parent
        is_child = any(child is self for _, _, child in new_parent.children)
        if is_child:
            self._parent = new_parent
        else:
            raise ValueError("Trying to set parent that doesn't have this node as child")


    @property
    def is_leaf_node(self):
        return self.children is None

    def print(self, prefix):
        if self.is_leaf_node:
            pass
        else:
            for lower, upper, child in self.children:

                print(f"{prefix} -  {self.bounds_to_split_str(lower, upper)}")
                child.print(prefix + '\t')

    def __hash__(self):
        return hash((str(self.instances), self.split_attribute_name))

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
        return copy.deepcopy(tree_node)

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
