"""Privileged-teacher work (sim fidelity, privileged expert, DAgger).

Kept apart from hardware/onboard on purpose: everything here only READS onboard/, imitation/
and the flight logs, and writes its own outputs, so deleting this folder changes nothing else.
"""
