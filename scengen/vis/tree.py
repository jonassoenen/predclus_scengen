import copy
from collections import deque
from typing import Tuple, Optional, List

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


class VisTree:
    def __init__(self, root_node, attribute_df, timeseries_df):
        self.root_node = root_node

        self.all_nodes = None
        self.all_leafs = None

        # self.root_node.check_validity()
        self.root_node.init_descriptions(parent_description=None)
        self.root_node.init_data(attribute_df, timeseries_df)
        self.set_node_ids()

        self.attribute_df = attribute_df
        self.timeseries_df = timeseries_df

    @classmethod
    def from_sklearn_tree(cls, sklearn_tree, attribute_df, timeseries_df):
        instances = np.arange(0, attribute_df.shape[0], 1)
        root_node = sklearn_tree_to_nodes(sklearn_tree, 0, instances, attribute_df)
        return VisTree(root_node, attribute_df, timeseries_df)

    def get_node(self, node_id):
        return self.all_nodes[node_id]

    def get_leaf(self, leaf_id):
        return self.all_leafs[leaf_id]

    def set_node_ids(self):
        # assign node ids in breath first fashion
        current_node_id = 0
        all_nodes = []
        current_leaf_id = 0
        all_leafs = []
        queue = deque([self.root_node])
        while len(queue) > 0:
            node = queue.popleft()
            if node.is_leaf_node:
                node.node_id = current_leaf_id
                all_leafs.append(node)
                current_leaf_id += 1
            else:
                node.node_id = current_node_id
                all_nodes.append(node)
                current_node_id += 1
                queue.extend(child for _, _, child in node.children)

        self.all_nodes = np.array(all_nodes, dtype='object')
        self.all_leafs = np.array(all_leafs, dtype='object')

    def plot_tree(self, figsize=(20, 25), max_depth = None):
        self.plot_subtree(self.root_node, figsize, max_depth)

    def plot_subtree(self, node, figsize=(20, 25), max_depth = None):
        # convert node id to node if required
        if isinstance(node, int):
            node = self.get_node(node)

        import matplotlib.pyplot as plt
        # figure configuration
        fig = plt.figure(figsize=figsize)
        if node is not self.root_node:
            plt.title(f"Subtree of node {node.node_id} with description {node.description}")
        else:
            plt.title(f"Full tree")

        plt.axis('off')

        # find all children in order
        leafs = []
        queue = deque([(node, 0)])
        while len(queue) > 0:
            node, depth = queue.pop()
            if (max_depth is not None and depth == max_depth) or node.is_leaf_node:
                leafs.append(node)
            # if node.is_leaf_node:
            #     leafs.append(node)
            else:
                queue.extend((child, depth+1) for _, _, child in node.children)

        x_pos = 400
        y_interval = 10
        x_interval = 30

        # first position the leaf nodes
        positions = {leaf: (x_pos, y) for leaf, y in zip(leafs, range(0, len(leafs) * y_interval + 1, y_interval))}

        # draw the leaf nodes
        for leaf, (x, y) in positions.items():
            if leaf.is_leaf_node:
                plt.text(x, y, leaf.node_id, horizontalalignment='center', verticalalignment='center',
                         bbox=dict(boxstyle="Circle, pad=0.35", facecolor='white'))
            else:
                plt.text(x,y, f'tree cut at node {leaf.node_id}', verticalalignment='center',
                     horizontalalignment='center', bbox={'facecolor': 'white', 'pad': 3})


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

        while len(queue) > 0:
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
            plt.text(x_min - x_interval, (y_min + y_max) / 2, f"{node_to_draw.split_attribute_name}",
                     verticalalignment='center',
                     horizontalalignment='center', bbox={'facecolor': 'white', 'pad': 3})

            # plot node id in gray
            plt.text(x_min - x_interval + 10, (y_min + y_max) / 2, node_to_draw.node_id, verticalalignment='center',
                     horizontalalignment='left', color='gray')

            # Save the position
            positions[node_to_draw] = (x_min - x_interval, (y_min + y_max) / 2)

            ## Check if parent becomes draweable
            if node_to_draw.parent is not None and all(
                    child in positions for _, _, child in node_to_draw.parent.children):
                queue.append(node_to_draw.parent)

    def print_tree(self):
        self.root_node.print(prefix='')

    def compress_tree(self):
        return VisTree(get_tree_with_compressed_splits(self.root_node), self.attribute_df, self.timeseries_df)

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
        self.description: Optional[str] = None
        self.node_id: Optional[int] = None

        self.attribute_df = None
        self.timeseries_df = None

    def init_descriptions(self, parent_description):

        self.description = "" if parent_description is None else parent_description

        if self.is_leaf_node:
            return
        for lower, upper, child in self.children:
            if parent_description is not None:
                description = parent_description + ' & ' + self.bounds_to_split_str(lower, upper)
            else:
                description = self.bounds_to_split_str(lower, upper)
            child.init_descriptions(description)
    def init_data(self, attribute_df, timeseries_df):
        self.attribute_df = attribute_df
        self.timeseries_df = timeseries_df
        if not self.is_leaf_node:
            for _, _, child in self.children:
                child.init_data(attribute_df, timeseries_df)

    def bounds_to_split_str(self, lower, upper, include_name=True):
        name = self.split_attribute_name if include_name else ' '
        if np.isinf(lower) and len(self.children) == 2:
            return f"{name} ≤ {upper:.2f}"
        elif np.isinf(upper) and len(self.children) == 2:
            return f"{name} > {lower:.2f}"
        else:
            return f"{lower:.2f} < {name} ≤ {upper:.2f}"

    @property
    def split_strings(self):
        for lower, upper, _ in self.children:
            yield self.bounds_to_split_str(lower, upper, include_name=False)

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
    def instance_timeseries_df(self):
        return self.timeseries_df.iloc[self.instances]

    @property
    def instance_attribute_df(self):
        return self.attribute_df.iloc[self.instances]

    @property
    def is_leaf_node(self):
        return self.children is None

    @property
    def node_name(self):
        if self.is_leaf_node:
            return f"Leaf {self.node_id}"
        else:
            return f"Node {self.node_id}"

    @property
    def nb_of_instances(self):
        return self.instances.shape[0]

    def print(self, prefix):
        if self.is_leaf_node:
            print(f'{prefix} ⤇ leaf {self.node_id}')
        else:
            for lower, upper, child in self.children:
                print(f"{prefix}({self.node_id}) {self.bounds_to_split_str(lower, upper)}")
                child.print(prefix + '\t')

    def _get_subtree_children_w_description(self, max_depth):
        queue = deque([(self, 0, None)])
        relevant_children = []
        while len(queue) > 0:
            node, current_depth, current_description = queue.pop()
            if current_depth == max_depth:
                relevant_children.append((node, current_description))
            elif current_depth < max_depth:
                for lower, upper, child in reversed(node.children):
                    description = node.bounds_to_split_str(lower, upper)
                    if current_description is not None:
                        description = current_description + ' & ' + description
                    queue.append((child, current_depth + 1, description))
        return relevant_children
    def plot_children_subtree(self, max_depth = 2, all_quantiles = False):
        # get relevant children
        relevant_children = self._get_subtree_children_w_description(max_depth)

        # plot charts
        charts = []
        for child, description in relevant_children:
            charts.append(
                child.plot_timeseries_quantiles(all = all_quantiles, raw = True).properties(title=f"{child.node_name}\n{description}")
            )
        return big_chart(alt.hconcat(*charts).resolve_scale(x='shared', y='shared', color='shared'))


    def plot_children(self, all_quantiles = False):
        charts = []
        for lower, upper, child in self.children:
            charts.append(
                child.plot_timeseries_quantiles(all = all_quantiles, raw = True).properties(title=f"{child.node_name}\n{self.bounds_to_split_str(lower, upper)}").interactive(bind_x = False))


        return big_chart(alt.hconcat(*charts).resolve_scale(x='shared', y='shared', color='shared'))

    def plot_timeseries(self, max_instances_to_show=10):
        nb_instances_to_show = min(max_instances_to_show, self.nb_of_instances)
        instances_to_show = np.random.choice(self.instances, nb_instances_to_show, replace = False)
        relevant_timeseries = self.timeseries_df.iloc[instances_to_show].reset_index(drop = True)
        plot_df = (
            relevant_timeseries
            .stack()
            .rename_axis(['timeseries', 'timestamp'], axis=0)
            .to_frame('value')
            .reset_index()
        )
        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X('timestamp', axis=alt.Axis(format="%H:%M")),
            y='value',
            color=alt.Color('timeseries:Q', legend=None)
        )
        node_text = 'node' if not self.is_leaf_node else 'leaf'
        return big_chart(chart.properties(title = f"Examples from {node_text} {self.node_id}", width = 500), grid = True)

    def plot_timeseries_quantiles(self, all = False, raw = False):
        data_df = self.instance_timeseries_df

        if all:
            q = np.arange(0, 1.01, 0.05)
        else:
            q = np.arange(0.05, 0.96, 0.05)
        quantile_df = data_df_to_quantiles(data_df, quantiles = q)
        chart =  (
            alt.Chart(quantile_df)
            .mark_area()
            .encode(
                x=alt.X("timestamp", axis=alt.Axis(format="%H:%M")),
                y=alt.Y(
                    "min:Q", title="consumption (kWh)", axis=alt.Axis(format=".2f")
                ),
                y2="max:Q",
                color="quantiles:O",
            )
        )
        node_text = 'node' if not self.is_leaf_node else 'leaf'
        if raw:
            return chart.properties(width = 500)
        return big_chart(chart.properties(title = f'Quantiles of TS in {node_text} {self.node_id}', width = 500), grid = True)



    def plot_feature_correlations(self, n=5, local=True):
        if local:
            data_df = self.instance_attribute_df
        else:
            data_df = self.attribute_df
        correlation_df = (
            data_df.corrwith(data_df[self.split_attribute_name], method='spearman')
            .drop(self.split_attribute_name)
            .sort_values(ascending=False)
            .iloc[:n]
        )
        # return correlation_df.to_frame('correlation').reset_index()

        correlation_plot = alt.Chart(correlation_df.to_frame('correlation').reset_index()).mark_bar().encode(
            y=alt.Y('index', title = None, sort=alt.EncodingSortField(field="Rank correlation", order='descending')),
            x=alt.X('correlation:Q', scale=alt.Scale(domain=[0, 1]))
        )
        return big_chart(correlation_plot.properties(title = f"Corr of {self.split_attribute_name} with alternative splits"), grid = False)

    def plot_nb_of_instances_per_child_subtree(self, max_depth = 2):
        relevant_children = self._get_subtree_children_w_description(max_depth)

        data_df = pd.DataFrame(columns=['nb_of_instances', 'lower_bound'])
        for idx, (child, description) in enumerate(relevant_children):
            data_df.loc[description] = child.nb_of_instances, idx

        bar_chart = alt.Chart(data_df.reset_index(), title='#Instances in child nodes').mark_bar().encode(
            y=alt.Y('index', title=None, sort=alt.EncodingSortField(field="lower_bound", order='ascending')),
            x='nb_of_instances'
        )

        text_chart = bar_chart.mark_text(align='left', baseline='middle', fontSize=20).encode(
            text='nb_of_instances'
        )

        return big_chart(alt.layer(bar_chart, text_chart))

    def plot_nb_of_instances_per_child(self):
        data_df = pd.DataFrame(columns = ['nb_of_instances', 'lower_bound'])
        for lower, upper, child in self.children:
            string = self.bounds_to_split_str(lower, upper, include_name = True)
            data_df.loc[string] = child.nb_of_instances, lower

        bar_chart = alt.Chart(data_df.reset_index(), title = '#Instances in child nodes').mark_bar().encode(
            y = alt.Y('index', title = None, sort=alt.EncodingSortField(field="lower_bound", order = 'ascending')),
            x = 'nb_of_instances'
        )

        text_chart = bar_chart.mark_text(align = 'left', baseline = 'middle', fontSize = 20).encode(
            text = 'nb_of_instances'
        )

        return big_chart(alt.layer(bar_chart, text_chart))
    def plot_attribute_distribution(self,  nb_of_bins = 'auto', bandwidth = None, kde = False, local=True):
        if local:
            data_points = self.instance_attribute_df.loc[:, self.split_attribute_name].to_numpy()
        else:
            data_points = self.attribute_df[self.split_attribute_name].to_numpy()
        if kde:
            # for continuous calculate kde
            kde = gaussian_kde(data_points, bw_method = bandwidth)
            min, max = np.min(data_points), np.max(data_points)
            plot_df = pd.DataFrame().assign(
                x=np.linspace(min, max, 100),
                kde=lambda df: kde(df.x)
            )
            dist_chart = alt.Chart(plot_df).mark_area().encode(
                x=alt.X('x', title=self.split_attribute_name),
                y='kde:Q'
            )
            text_height = plot_df.kde.max()
        else:
            # make histogram
            values, boundaries = np.histogram(data_points, bins= nb_of_bins, density = False)
            windows = np.lib.stride_tricks.sliding_window_view(boundaries, 2)
            plot_df = pd.DataFrame(windows, columns = ['lower_bound', 'upper_bound']).assign(value = values)
            dist_chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('lower_bound', bin='binned', title = self.split_attribute_name),
                x2='upper_bound',
                y=alt.Y('value:Q', title = 'count')
            )
            text_height = values.max()


        bounds = set()
        for lower, upper, _ in self.children:
            if not np.isinf(lower):
                bounds.add(lower)
            if not np.isinf(upper):
                bounds.add(upper)

        bounds_df = pd.DataFrame(dict(threshold=list(bounds)))

        vline_chart = alt.Chart(bounds_df).mark_rule().encode(
            x='threshold'
        )

        text_chart = alt.Chart(bounds_df).mark_text(angle  = 270, baseline = 'top', align = 'right', dy = 5, fontSize = 15).encode(

            x = 'threshold',
            y = alt.YValue(text_height*0.9),
            text = alt.Text('threshold', format = '.2f'),
        )

        return big_chart(alt.layer(dist_chart, vline_chart).properties(title = f"Split of node {self.node_id} on {self.split_attribute_name}"), grid = True)


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
        return copy.copy(tree_node)

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


def sklearn_tree_to_nodes(tree, node_idx, instances, attribute_df):
    feature_idx = tree.feature[node_idx]
    feature_threshold = tree.threshold[node_idx]

    is_leaf_node = tree.children_left[node_idx] == tree.children_right[node_idx]
    if is_leaf_node:
        return Node(instances, None, None)
    else:
        test = attribute_df.iloc[:, feature_idx] <= feature_threshold
        left_attributes = attribute_df[test]
        left_instances = instances[test]
        left_node_idx = tree.children_left[node_idx]
        left_subtree = sklearn_tree_to_nodes(tree, left_node_idx, left_instances, left_attributes)

        right_attributes = attribute_df[~test].copy()
        right_instances = instances[~test]
        right_node_idx = tree.children_right[node_idx]
        right_subtree = sklearn_tree_to_nodes(tree, right_node_idx, right_instances, right_attributes)

        children = [
            (-np.infty, feature_threshold, left_subtree),
            (feature_threshold, np.infty, right_subtree)
        ]
        return Node(instances, children, attribute_df.columns[feature_idx])


def data_df_to_quantiles(data_df, quantiles=None):
    # q = np.concatenate([np.arange(0, 0.05, 0.01), np.arange(0.05, 0.96, 0.05), np.arange(0.95, 1.005, 0.01)])
    if quantiles is None:
        q = np.arange(0, 1.01, 0.05)
    else:
        q = quantiles

    quantiles = data_df.quantile(q, interpolation="nearest").set_axis(
        (q * 100).astype("int"), axis=0
    )
    lower_quantiles = (
        quantiles.loc[:49, :]
        .stack()
        .to_frame("min")
        .rename_axis(index=["lower_quantile", "timestamp"])
    )
    upper_quantiles = (
        quantiles.loc[51:, :]
        .stack()
        .to_frame("max")
        .reset_index(level=0)
        .rename(columns={"level_0": "upper_quantile"})
        .assign(lower_quantile=lambda x: 100 - x.upper_quantile)
        .set_index("lower_quantile", append=True)
        .swaplevel(0, 1)
        .sort_index()
        .rename_axis(index=["lower_quantile", "timestamp"])
    )
    return (
        lower_quantiles.join(upper_quantiles)
        .reset_index()
        .assign(
            quantiles=lambda x: x.lower_quantile.astype("str").str.zfill(2)
                                + "-"
                                + x.upper_quantile.astype("str").str.zfill(2)
        )
        .drop(columns=["lower_quantile", "upper_quantile"])
    )

def big_chart(chart, fontsize=20, grid = False):
    return (
        chart.configure_axis(
            grid=grid,
            labelFontSize=fontsize,
            titleFontSize=fontsize,
            offset=5,
            labelLimit = 300
        )
        .configure_title(fontSize=fontsize)
        .configure_legend(titleFontSize=fontsize, labelFontSize=fontsize)
        .configure_view(strokeWidth=0)
    )
