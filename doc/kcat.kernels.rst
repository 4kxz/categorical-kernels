kcat.kernels package
====================

This package contains all the kernel related modules.

The core of the package is in the :mod:`kcat.kernels.functions` module
which contains the kernel functions.

The :mod:`kcat.kernels.search` and :mod:`kcat.kernels.models`
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


kcat.kernels.search module
-------------------------------

.. automodule:: kcat.kernels.search

.. autoclass:: kcat.kernels.search.SearchBase
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.SearchELK
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.SearchK0
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.SearchK1
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.SearchK2
    :members:
    :exclude-members: __doc__, __module__

.. autoclass:: kcat.kernels.search.SearchM1
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
