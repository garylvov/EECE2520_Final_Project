import sympy as sp
import matplotlib
from matplotlib import pyplot as plt

from signals import Signal as x
from signals import UnitStep as u
from signals import RectangularPulse as rect
from signals import TriangularPulse as tri
from signals import Ramp as r


t = sp.symbols('t')

signal_to_proc = rect(t)
fig = plt.figure(figsize=(20, 20))

signal_plot = fig.add_subplot(2, 1, 1)
fourier_plot = fig.add_subplot(2, 1, 2)
signal_to_proc.plot_discretized_signal(signal_plot)
signal_to_proc.plot_discretized_fourier_transform(fourier_plot)

signal_plot.legend(fontsize="20")
fourier_plot.legend(fontsize="20")
plt.show()