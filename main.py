import sympy as sp
from matplotlib import pyplot as plt

from signals import Signal as x
from signals import UnitStep as u
from signals import RectangularPulse as rect
from signals import TriangularPulse as tri
from signals import Ramp as r

plt.rcParams['text.usetex'] = True

t = sp.symbols('t')
signal_to_proc = rect(t - 4)
fig = plt.figure(figsize=(20, 20))

signal_plot = fig.add_subplot(2, 1, 1)
fourier_plot = fig.add_subplot(2, 1, 2)
signal_to_proc.plot_discretized_signal(signal_plot)
signal_to_proc.plot_discretized_fourier_transform(fourier_plot)

# Show the plot
fourier_plot.legend()
plt.show()