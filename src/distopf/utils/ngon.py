import numpy as np


def ngon_line_equations(n, theta0=0, decimals=12):
    """
    Generate line equations ax + by = 1 for a regular n-gon
    inscribed in a unit circle.

    Returns:
        List of (a, b) tuples of Python floats.
    """
    r = 1.0 / np.cos(np.pi / n)
    angles = theta0 + np.pi / n + np.arange(n) * (2 * np.pi / n)
    eqs = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    eqs = np.round(eqs, decimals) + 0.0
    return [tuple(row) for row in eqs.tolist()]


def print_table(n, theta0=0):
    """
    Print a formatted table of line equations for an n-gon.
    """
    lines = ngon_line_equations(n, theta0)

    print(f"Regular {n}-gon inscribed in unit circle")
    print(f"Starting angle: {np.degrees(theta0):.1f}°")
    print()
    print(f"{'Line':<6} {'a':>10} {'b':>10}")
    print("-" * 40)

    for i, (a, b) in enumerate(lines):
        print(f"{i + 1:<6} {a:>10.6f} {b:>10.6f}")

    print()


def get_ngon_info(n, theta0=0):
    """
    Print additional information about the n-gon.
    """
    side_length = 2 * np.sin(np.pi / n)
    apothem = np.cos(np.pi / n)
    area = 0.5 * n * side_length * apothem
    circle_area = np.pi
    percent_of_circle = (area / circle_area) * 100
    print(f"Percent of circle area: {percent_of_circle:.2f}%")
    print(f"Side length: {side_length:.6f}")
    print(f"Apothem (distance from center to side): {apothem:.6f}")
    import time

    start_time = time.perf_counter()
    ngon_line_equations(n, theta0)
    time_to_compute = time.perf_counter() - start_time
    print(f"time to compute: {time_to_compute:.6f} seconds")


# Example usage
if __name__ == "__main__":
    # Square with sides parallel to axes
    print("=" * 50)
    print_table(4, theta0=0)
    get_ngon_info(4, theta0=0)
    # Hexagon with vertex on x-axis
    print("=" * 50)
    print_table(6, theta0=0)
    get_ngon_info(6, theta0=0)

    # Octagon with vertex on x-axis
    print("=" * 50)
    print_table(8, theta0=0)
    get_ngon_info(8, theta0=0)
    # Octagon with vertex on x-axis
    print("=" * 50)
    print_table(16, theta0=0)
    get_ngon_info(16, theta0=0)
    print("=" * 50)
    get_ngon_info(1e3, theta0=0)
