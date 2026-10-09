"""
Onboard-camera hoop traversal: fly the real Tello through a blue ring using only its own
front camera, and record the flights in a form that behaviour cloning and RL can train on.

Separate from hardware/common/ and hardware/state_estimation/ on purpose. Those implement
the simulator's task (reach a point in a cage, position from an overhead ArUco camera).
This implements a different task (pass through a ring, position implicit in what the
front camera sees) and shares nothing with them except the repository.

Read spec.py first: it defines the observation and action every other module here uses.
"""
