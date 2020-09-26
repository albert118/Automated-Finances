Testing Documentation
*********************

Running The Tests
^^^^^^^^^^^^^^^^^

Prior to running any tests, initiate the virtual environment, if you haven't set that up
yet checkout the setup :doc:`setup`.

**Windows**

`> Scripts\activate`

**Unix**

`> source Source\activate`

In the top directory, run the following command to run all the tests.

`python -m unittest discover -s test -p "*_test.py"`

Account Data Testing
^^^^^^^^^^^^^^^^^^^^

.. module:: accountdata

Transaction Data Testing
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TxData
	:members:
