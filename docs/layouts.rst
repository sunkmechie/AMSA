Layouts
=======

A :class:`amsa.layouts.MVLayout` describes how multivector coefficients are ordered and which blades are present. It is separate from both the algebra semantics and the storage backend.

Layout kinds
------------

Dense layout
^^^^^^^^^^^^

Covers the full algebra basis in canonical blade order:

.. code-block:: python

   layout = MVLayout.dense(algebra.spec)

Grade layout
^^^^^^^^^^^^

Contains only blades of the chosen grades, packed in canonical order:

.. code-block:: python

   layout = MVLayout.grade(algebra.spec, 1, 2)

Sparse layout
^^^^^^^^^^^^^

Arbitrary blade subsets. Empty sparse layouts are allowed:

.. code-block:: python

   layout = MVLayout.sparse_pattern(algebra.spec, (1, 3), name="support")

Layout metadata
---------------

- ``blades`` — tuple of blade bit-patterns in the layout
- ``kind`` — ``"dense"``, ``"grade"``, or ``"sparse"``
- ``name`` — human-readable label
- ``size`` — number of coefficients per multivector
- ``grades`` — sorted unique grades present in the layout
- ``blade_names()`` — canonical names for the layout's blades
- ``index_of(blade)`` — coefficient index of a blade
- ``contains(blade)`` — membership test
