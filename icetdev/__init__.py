from __future__ import print_function, division
from .structure import Structure
from .cluster_space import ClusterSpace
from .cluster_expansion import ClusterExpansion
from .structure_container import StructureContainer
from .fitting import Optimizer

'''
icet module
'''

__description__ = 'The pythonic approach to cluster expansions'
__authors__ = ['Mattias Ångqvist',
               'William Armando Muñoz',
               'Thomas Holm Rod',
               'Paul Erhart']
__copyright__ = ''
__license__ = ''
__credits__ = ['Mattias Ångqvist',
               'William Armando Muñoz',
               'Thomas Holm Rod',
               'Paul Erhart']
__version__ = '0.1'
__all__ = ['ClusterSpace',
           'ClusterExpansion',
           'StructureContainer',
           'Structure',
           'Optimizer']
__status__ = 'beta-version'
__url__ = 'http://www.icet.org/'
