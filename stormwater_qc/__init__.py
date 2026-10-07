"""stormwater_qc - quality control of stormwater infrastructure data for SWMM.

Modules, in the order they are normally applied:

* ``geometry``   - snap near-miss pipe ends, split pipes into segments
* ``networks``   - identify, classify and clean independent networks; duplicates
* ``topology``   - junctions, flow direction toward outfalls, connectivity
* ``inlets``     - select, snap, cluster and merge inlets
* ``diameters``  - implausible and missing diameters
* ``slopes``     - DEM slopes within full-flow velocity limits
* ``watersheds`` - one catch basin per inlet cluster (needs pysheds)
* ``swmm``       - junction depths, inverts and conduit offsets
* ``synthetic``  - synthetic network with all these defects, for testing
"""
__version__ = "1.0.0"
