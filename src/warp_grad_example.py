import warp as wp


@wp.kernel
def add(term_1: wp.array(dtype=float), term_2: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()

    result[tid] = term_1[tid] + term_2[tid]


tape = wp.Tape()

a = wp.array([2.0], dtype=wp.float32)
b = wp.array([3.0], dtype=wp.float32, requires_grad=True)
c = wp.array([4.0], dtype=wp.float32)
d = c
e = wp.array([5.0], dtype=wp.float32, requires_grad=True)

result = wp.zeros(1, dtype=wp.float32, requires_grad=True)

with tape:
    wp.launch(add, dim=1, inputs=[b, e], outputs=[a])

    # ScopedTimer registers itself as a scope on the tape
    with wp.ScopedTimer("Adder"):

        # we can also manually record scopes
        tape.record_scope_begin("Custom Scope")
        wp.launch(add, dim=1, inputs=[a, b], outputs=[c])
        tape.record_scope_end()

        wp.launch(add, dim=1, inputs=[d, a], outputs=[result])


tape.visualize(
    filename="tape.dot",
    array_labels={a: "a", b: "b", c: "c", e: "e", result: "result"},
)