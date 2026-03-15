"""Smoke tests — quick sanity checks that don't require GPU."""

import pytest


def test_import_memory_maze():
    import memory_maze  # verifies package loads without error


def test_import_train_impala():
    from train_impala import MemoryMazeNet
    assert MemoryMazeNet is not None


def test_import_train_dreamer():
    from train_dreamer import DreamerModel
    assert DreamerModel is not None


def test_import_torchbeast():
    from torchbeast.core import vtrace
    from torchbeast.core import environment
    from torchbeast.core import file_writer
    from torchbeast.core import prof
    assert vtrace is not None
