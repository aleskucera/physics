"""2D ball throw example.

This example demonstrates how to optimize the initial velocity of the ball in order to hit a target.
"""

from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def loss_fn(trajectory: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
    distances = torch.sum((trajectory - target) ** 2, dim=1)
    min_distance, min_index = distances.min(0)
    return min_distance, min_index


@dataclass
class Ball:
    pos: torch.Tensor
    vel: torch.Tensor
    mass: float
    g: torch.Tensor = torch.tensor([0.0, -9.81])

    def sim_step(self, dt: float) -> torch.Tensor:
        """Update the ball state."""
        self.pos = self.pos + self.vel * dt
        self.vel = self.vel + self.g * dt

        return self.pos


class BallModel(nn.Module):
    def __init__(self, ball_initial_velocity: torch.Tensor, dt: float = 0.01, sim_steps: int = 1000):
        super().__init__()

        # Initial velocity is a parameter so that it can be optimized
        self.ball_initial_velocity = nn.Parameter(ball_initial_velocity)

        # Simulation parameters
        self.dt = dt
        self.sim_steps = sim_steps

    def forward(self, ball_mass: float, ball_initial_position: torch.Tensor) -> torch.Tensor:
        ball = Ball(pos=ball_initial_position, 
                    vel=self.ball_initial_velocity, 
                    mass=ball_mass)
        trajectory = torch.empty((self.sim_steps, 2), dtype=torch.float32)

        trajectory[0] = ball.pos
        for i in range(1, self.sim_steps):
            trajectory[i] = ball.sim_step(self.dt)

        return trajectory

@torch.no_grad()
def plot_trajectory(ax, ball_initial_position: torch.Tensor, ball_target_position: torch.Tensor, trajectory: torch.Tensor):
    """Plot the ball trajectory."""
    ax.clear()
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Trajectory")
    ax.plot(ball_initial_position[0], ball_initial_position[1], "go", label="Start")
    ax.plot(ball_target_position[0], ball_target_position[1], "ro", label="Target")
    ax.legend()
    plt.draw()
    plt.pause(0.01)

@torch.no_grad()
def animate_trajectory(ball_initial_position: torch.Tensor, ball_target_position: torch.Tensor, trajectory: torch.Tensor):
    """Animate the ball trajectory."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 15)
    ax.set_ylim(-1, 10)
    line, = ax.plot([], [], "b-", label="Trajectory")
    start_dot, = ax.plot([], [], "go", label="Start")
    target_dot, = ax.plot([], [], "ro", label="Target")
    start_dot.set_data(ball_initial_position[0], ball_initial_position[1])
    target_dot.set_data(ball_target_position[0], ball_target_position[1])

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
        return line,

    ani = FuncAnimation(fig, update, frames=len(trajectory), init_func=init, blit=True, repeat=False)
    plt.legend()
    plt.show()

    
    
def main():
    verbose = True

    ball_mass = 0.2
    ball_initial_position = torch.tensor([0.0, 5.0])
    ball_initial_velocity = torch.tensor([5.0, 5.0])
    ball_target = torch.tensor([10.0, 0.0])

    num_epochs = 100
    learning_rate = 0.015

    model = BallModel(ball_initial_velocity=ball_initial_velocity,
                      dt=0.01, 
                      sim_steps=1000)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    plt.ion()
    fig, ax = plt.subplots()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        trajectory = model(ball_mass, ball_initial_position)
        loss, min_dist_index = loss_fn(trajectory, ball_target)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
            plot_trajectory(ax, ball_initial_position, ball_target, trajectory[:min_dist_index+1])
        
        loss.backward()
        optimizer.step()

    # Disable interactive mode and show the final plot
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
    
