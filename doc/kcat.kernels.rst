kcat.kernels package
====================

This package contains all the kernel related modules.

The core of the package is in the :mod:`kcat.kernels.functions` module
which contains the kernel functions.

The :mod:`kcat.kernels.search` and :mod:`kcat.kernels.helpers`
modules contain additional code to use the kernels with the scikit-learn
environment and run tests.


kcat.kernels.functions module
-------------------------------

.. automodule:: kcat.kernels.functions

.. autofunction:: kcat.kernels.functions.elk

.. autofunction:: kcat.kernels.functions.k0

.. autofunction:: kcat.kernels.functions.k1

.. autofunction:: kcat.kernels.functions.k2

.. autofunction:: kcat.kernels.functions.m3


kcat.kernels.search module
-------------------------------

.. automodule:: kcat.kernels.search

.. autoclass:: kcat.kernels.search.BaseSearch
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.ELKSearch
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.K0Search
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.K1Search
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.K2Search
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.M3Search
    :members:
    :exclude-members: __doc__, __module__


kcat.kernels.helpers module
-------------------------------

.. automodule:: kcat.kernels.helpers

.. autoclass:: kcat.kernels.helpers.BaseHelper
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.helpers.ELK
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.helpers.K0
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.helpers.K1
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.helpers.K2
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.helpers.M3
    :members:
    :undoc-members:
    :exclude-members: __doc__, __module__, kernel, searcher

.. autoclass:: kcat.kernels.helpers.RBF
    :exclude-members: __doc__, __module__, kernel, searcher
    :members:
    :undoc-members:
