import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

s = sp.symbols('s')
F = (s + 1)/(s**2 + 2*s + 1)

F_apart = sp.apart(F, s)
poles = sp.roots(sp.denom(F_apart), s)
zeros = sp.roots(sp.numer(F_apart), s)

# Create numerical functions for real and imaginary parts of s
re_func = sp.lambdify(s, sp.re(s))
im_func = sp.lambdify(s, sp.im(s))

fig, ax = plt.subplots()
ax.plot(re_func(s), im_func(s), color='blue')

# Shade the region of convergence based on pole positions
pole_colors = ['red', 'green', 'blue']  # Colors for shading based on pole positions
for idx, pole in enumerate(poles.keys()):
    ax.plot(re_func(pole), im_func(pole), marker='x', markersize=8, color=pole_colors[idx % len(pole_colors)])

    # Shade the region of convergence
    shade_color = pole_colors[idx % len(pole_colors)]
    convergence_region = sp.Interval(-sp.oo, sp.re(pole))
    polygon_vertices = [(float(convergence_region.start), ax.get_ylim()[0]),
                        (float(convergence_region.start), ax.get_ylim()[1]),
                        (float(convergence_region.end), ax.get_ylim()[1]),
                        (float(convergence_region.end), ax.get_ylim()[0])]
    polygon = Polygon(polygon_vertices, closed=True, alpha=0.2, edgecolor='none', facecolor=shade_color)
    ax.add_patch(polygon)

for zero in zeros.keys():
    ax.plot(re_func(zero), im_func(zero), marker='o', markersize=8, color='green')

ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_title('Pole-Zero Plot')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax.grid(True)
plt.show()

convergence_intervals = sp.Union(*[sp.Interval.open(-sp.oo, sp.re(pole)) for pole in poles.keys()])
print("Region of Convergence: ", convergence_intervals)
