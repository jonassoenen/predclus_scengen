import unittest
from scengen.vis.tree import VisTree, Node
import numpy as np

def test_equality_single_node():
    parent = Node(np.arange(1, 10), None, 'test1')
    tree1 = VisTree(parent)

    parent = Node(np.arange(1, 10), None, 'test1')
    tree2 = VisTree(parent)

    assert tree1 == tree2

def test_inequality_single_node():
    parent = Node(np.arange(1, 10), None, 'test1')
    tree1 = VisTree(parent)

    parent = Node(np.arange(1, 9), None, 'test1')
    tree2 = VisTree(parent)

    assert tree1 != tree2

    parent = Node(np.arange(1, 10), None, 'test1')
    tree1 = VisTree(parent)

    parent = Node(np.arange(1, 10), None, 'test2')
    tree2 = VisTree(parent)

    assert tree1 == tree2

def test_equality_small_tree():
    left_child = Node(np.arange(1,5), None, None)
    right_child = Node(np.arange(5,10), None, None)
    parent = Node(np.arange(1, 10), [(-np.infty, 5, left_child), (5, np.infty, right_child)], 'test1')
    tree1 = VisTree(parent)


    left_child = Node(np.arange(1,5), None, None)
    right_child = Node(np.arange(5,10), None, None)
    parent = Node(np.arange(1, 10), [(-np.infty, 5, left_child), (5, np.infty, right_child)], 'test1')
    tree2 = VisTree(parent)
    assert tree1 == tree2

def test_inequality_small_tree():
    left_child = Node(np.arange(1,6), None, None)
    right_child = Node(np.arange(6,10), None, None)
    parent = Node(np.arange(1, 10), [(-np.infty, 5, left_child), (5, np.infty, right_child)], 'test1')
    tree1 = VisTree(parent)


    left_child = Node(np.arange(1,5), None, None)
    right_child = Node(np.arange(5,10), None, None)
    parent = Node(np.arange(1, 10), [(-np.infty, 5, left_child), (5, np.infty, right_child)], 'test1')
    tree2 = VisTree(parent)
    assert tree1 != tree2
def test_compress_single_tree():
    child = [Node(np.array([i]), None, None) for i in range(4)]
    first_node = Node(
        np.array([0,1]),
        [
            (-np.infty, 5, child[0]),
            (5, np.infty, child[1]),
        ],
        'A'
    )
    second_node = Node(
        np.array([0, 1]),
        [
            (-np.infty, 5, child[2]),
            (5, np.infty, child[3]),
        ],
        'B'
    )
    root_node = Node(
        np.arange(0,4),
        [
            (-np.infty, 10, first_node),
            (10, np.infty, second_node)
        ],
        'A'
    )
    tree = VisTree(root_node)
    compressed_tree = tree.compress_tree()

    b_node = second_node
    new_root_node = Node(
        np.arange(0,4),
        [
            (-np.infty, 5, child[0]),
            (5, 10, child[1]),
            (10, np.infty, b_node)
        ],
        'A'
    )
    correct_compressed_tree = VisTree(new_root_node)
    assert compressed_tree == correct_compressed_tree

def test_print_tree():
    child = [Node(np.array([i]), None, None) for i in range(4)]
    first_node = Node(
        np.array([0,1]),
        [
            (-np.infty, 5, child[0]),
            (5, np.infty, child[1]),
        ],
        'A'
    )
    second_node = Node(
        np.array([0, 1]),
        [
            (-np.infty, 5, child[2]),
            (5, np.infty, child[3]),
        ],
        'B'
    )
    root_node = Node(
        np.arange(0,4),
        [
            (-np.infty, 10, first_node),
            (10, np.infty, second_node)
        ],
        'A'
    )
    tree = VisTree(root_node)
    print()
    tree.print_tree()

    print()
    tree.compress_tree().print_tree()

