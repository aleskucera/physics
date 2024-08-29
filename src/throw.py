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

def loss_fn(trajectory: torch.Tensor, optimal_trajectory: torch.Tensor) -> torch.Tensor:
    return torch.sum((trajectory - optimal_trajectory) ** 2)


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
def plot_trajectory(ax, ball_initial_position: torch.Tensor, trajectory: torch.Tensor, optimal_trajectory: torch.Tensor):
    """Plot the ball trajectory."""
    ax.clear()
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Trajectory")
    ax.plot(optimal_trajectory[:, 0], optimal_trajectory[:, 1], "r-", label="Optimal Trajectory")
    ax.plot(ball_initial_position[0], ball_initial_position[1], "go", label="Start")
    ax.legend()
    plt.draw()
    plt.pause(0.01)
    
    
def main():
    verbose = True

    ball_mass = 0.2
    ball_initial_position = torch.tensor([0.0, 5.0])
    ball_initial_velocity = torch.tensor([5.0, 5.0])
    ball_optimal_velocity = torch.tensor([6.0, 7.0])
    # ball_target = torch.tensor([10.0, 0.0])

    num_epochs = 100
    learning_rate = 0.04

    optimal_model = BallModel(ball_initial_velocity=ball_optimal_velocity,
                      dt=0.01, 
                      sim_steps=200)
    
    optimal_trajectory = optimal_model(ball_mass, ball_initial_position)

    # Add gauss noise
    noise = torch.normal(mean=0.0, std=0.03, size=optimal_trajectory.shape)
    optimal_trajectory = optimal_trajectory + noise

    model = BallModel(ball_initial_velocity=ball_initial_velocity,
                      dt=0.01, 
                      sim_steps=200)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    plt.ion()
    fig, ax = plt.subplots()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        trajectory = model(ball_mass, ball_initial_position)
        loss = loss_fn(trajectory, optimal_trajectory)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
            plot_trajectory(ax, ball_initial_position, trajectory, optimal_trajectory)
        
        loss.backward(retain_graph=True)
        optimizer.step()

    # Disable interactive mode and show the final plot
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
    
