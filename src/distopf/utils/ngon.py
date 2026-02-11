import numpy as np


def ngon_line_equations(n, theta0=0):
    """
    Generate line equations for a regular n-gon inscribed in a unit circle.

    Parameters:
        n      : number of sides
        theta0 : starting angle in radians (where first vertex is located)

    Returns:
        List of tuples (a, b) for equations ax + by = 1
    """
    # Calculate vertex angles
    angles = [theta0 + k * (2 * np.pi / n) for k in range(n)]
    # Calculate vertex coordinates
    vertices = [(np.cos(angle), np.sin(angle)) for angle in angles]
    # Calculate line equations for each side
    lines = []
    for k in range(n):
        # Get consecutive vertices
        x1, y1 = vertices[k]
        x2, y2 = vertices[(k + 1) % n]
        # Calculate coefficients
        a = y1 - y2
        b = x2 - x1
        c = a * x1 + b * y1
        lines.append([a / c, b / c])
    return np.array(lines)


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

    print(f"Side length: {side_length:.6f}")
    print(f"Apothem (distance from center to side): {apothem:.6f}")
    print()


# Example usage
if __name__ == "__main__":
    # Square with sides parallel to axes
    print("=" * 50)
    print_table(4, theta0=0)

    # Hexagon with vertex on x-axis
    print("=" * 50)
    print_table(6, theta0=0)

    # Octagon with vertex on x-axis
    print("=" * 50)
    print_table(8, theta0=0)
    # Octagon with vertex on x-axis
    print("=" * 50)
    print_table(16, theta0=0)
