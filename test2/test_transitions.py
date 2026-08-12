import numpy as np
from transitions import make_transition


# Temporary test sequences
a = np.zeros((40, 135, 2))
b = np.ones((30, 135, 2))

transition = make_transition(a, b, frames=15)

print("A:", a.shape)
print("B:", b.shape)
print("Transition:", transition.shape)

print()
print("First transition frame:")
print(transition[0, 0])

print()
print("Last transition frame:")
print(transition[-1, 0])