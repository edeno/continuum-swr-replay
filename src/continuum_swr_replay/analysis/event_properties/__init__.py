"""What the non-local events are like, and how SWR and non-SWR events differ.

- ``event_statistics``: the summary statistics of the immobile non-local
  events behind the Part 2 summary.
- ``property_contrast``: the nested bootstrap of the SWR against non-SWR
  property contrasts.
- ``class_or_continuum``: whether SWR-coupled and non-SWR events differ by
  kind or by degree of ripple power.
- ``event_analysis``: event intervals, durations and the immobile events.
- ``triggered_analysis``: signal windows around events and their averages.
- ``brain_state_reference``: the running and quiet-rest theta, ripple and MUA
  references an event's levels are read against.
- ``gamma_power``: slow- and fast-gamma power on the decode grid.
- ``start_and_rest_theta``: each event's start distance and the theta of
  duration-matched quiet rest.
- ``content_occupancy``: each content bin's track distance to the animal and
  to the nearest reward well.

Imports ``event_definition``, ``detection``, ``decode_quality`` and
``event_evidence``. Submodules are imported by name; nothing is re-exported
here.
"""
