import torch

# Define initial state and target
num_states = 4
initial_state = torch.tensor([1.0, 2.0], requires_grad=True)
target = torch.tensor([5.0, 10.0])

# Initialize states tensor
states = torch.zeros(num_states, 2)
states[0] = initial_state

# Generate states iteratively
for i in range(1, num_states):
    states[i] = states[i - 1] * 2

# Find the distances of each state from the target
distances = torch.sum((states - target) ** 2, dim=1)

# Identify the closest state and its index
min_distance, min_index = distances.min(0)

# Compute loss only for the closest state
loss = distances[min_index]
loss.backward()

# Print results
print("States:\n", states)
print("Initial State Gradient:\n", initial_state.grad)
