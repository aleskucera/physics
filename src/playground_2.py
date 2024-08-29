import warp as wp

@wp.kernel
def multiply_vec_by_scalar(vec: wp.array(dtype=wp.vec2),
                 scalar: wp.float32,
                 result: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    result[tid] = vec[tid] * scalar

@wp.kernel
def compute_loss(state: wp.array(dtype=wp.vec2),
                 target: wp.array(dtype=wp.vec2),
                 result: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    result[tid] = wp.dot((state[tid] - target[tid]), (state[tid] - target[tid]))

def main():
    num_states = 4

    # Initial state is vector that requires gradients
    initial_state = wp.array([1.0, 2.0], dtype=wp.vec2, requires_grad=True)
    target = wp.array([5.0, 10.0], dtype=wp.vec2)
    scalar = wp.float32(3.0)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    states = [wp.zeros(1, dtype=wp.vec2, requires_grad=True) for _ in range(num_states)]

    tape = wp.Tape()

    states[0] = initial_state
    with tape:
        for i in range(1, num_states):
            wp.launch(multiply_vec_by_scalar, 
                                    dim=1, 
                                    inputs=[states[i - 1], scalar], 
                                    outputs=[states[i]])

        wp.launch(compute_loss, 
                            dim=1, 
                            inputs=[states[2], target], 
                            outputs=[loss])

    tape.visualize(
        filename="my_tape.dot", 
        array_labels={
            initial_state: "initial_state",
            target: "target",
            scalar: "scalar",
            states[0]: "states[0]",
            states[1]: "states[1]",
            states[2]: "states[2]",
            states[3]: "states[3]",
        }
    )

    # Compute gradient
    tape.backward(loss)

    # Print results
    print("Initial State Gradient:\n", initial_state.grad)
    print("Target Gradient:\n", target.grad)

    for i in range(num_states):
        print(states[i])

    

if __name__ == "__main__":
    main()