Quadracheer
=====

A python library for singular and almost singular integrals. Especially those
integrals that are necessary for boundary element computations.

Functions return the points and weights of quadrature formulas defined on
the reference interval [-1, 1].

Installation
-----

If you've downloaded the code, just run::

    $ python setup.py install 

If you prefer to use pip, try::
    
    $ pip install quadracheer

Tests
-----

To run the tests, type::

    $ ./pytest
    
from the root directory. This assumes that py.test is installed. 
