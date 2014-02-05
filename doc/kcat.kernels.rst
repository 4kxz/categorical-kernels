kcat.kernels package
====================

This package contains all the kernel related modules.

The core of the package is in the :mod:`kcat.kernels.functions` module
which contains the kernel functions.

The :mod:`kcat.kernels.grid_search` and :mod:`kcat.kernels.models`
modules contain additional code to use the kernels with the scikit-learn
environment and run tests.


kcat.kernels.functions module
-------------------------------

.. automodule:: kcat.kernels.functions

.. autofunction:: kcat.kernels.functions.elk

.. autofunction:: kcat.kernels.functions.k0

.. autofunction:: kcat.kernels.functions.k1

.. autofunction:: kcat.kernels.functions.k2

.. autofunction:: kcat.kernels.functions.m1


kcat.kernels.grid_search module
-------------------------------

.. automodule:: kcat.kernels.grid_search

.. autoclass:: kcat.kernels.grid_search.GridSearchWrapper
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.grid_search.GridSearchELK
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.grid_search.GridSearchK0
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.grid_search.GridSearchK1
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.grid_search.GridSearchK2
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.grid_search.GridSearchM1
    :members:
    :exclude-members: __doc__, __module__


kcat.kernels.models module
-------------------------------

.. automodule:: kcat.kernels.models

.. autoclass:: kcat.kernels.models.Model
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.models.ELK
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.models.K0
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.models.K1
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.models.K2
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.models.M1
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.models.RBF
    :exclude-members: __doc__, __module__, kernel, searcher
    :members:
    :undoc-members:
