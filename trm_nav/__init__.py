"""
NanoNav: Tiny Recursive Model for Local Robot Navigation

A proof-of-concept implementation comparing TRM to A* for grid navigation.
"""

__version__ = "0.1.0"

from .a_star import astar_path
from .map_generator import generate_random_map, generate_solvable_map
from .dataset import NavigationDataset, build_dataset
from .model import TRMNavigator
