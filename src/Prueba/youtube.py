from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

############################################################
# Mas simplre sin Aer

from qiskit.quantum_info import Statevector

qc = QuantumCircuit(1)
# initial_state = [0, 1] # Estado |1>
initial_state = [1/2**0.5, 1/2**0.5] # Estado de superposición |+> = (1/√2)(|0> + |1>)
# initial_state = [1, 0] # Estado |0>
qc.initialize(initial_state, 0)

statevector = Statevector.from_instruction(qc)

plot_bloch_multivector(statevector)
plt.show()
############################################################

# Usando Aer pero sin puertas adicionales

# qc = QuantumCircuit(1)
# initial_state = [0, 1]
# qc.initialize(initial_state, 0)

# qc.save_statevector()  # <-- agregar esto

# backend = Aer.get_backend('aer_simulator')
# qc_t = transpile(qc, backend)
# result = backend.run(qc_t).result()
# statevector = result.get_statevector(0)  # o result.data(0)['statevector']

# plot_bloch_multivector(statevector)
# plt.show()


############################################################

# Usando Aer con puertas adicionales

# qc = QuantumCircuit(1)
# initial_state = [0, 1] # Estado |1>
# qc.initialize(initial_state, 0)

# qc.x(0) # Aplicar una puerta X(NOT)
# qc.h(0) # Aplicar una puerta Hadamard(superposición)

# qc.save_statevector()  # <-- agregar esto
# backend = Aer.get_backend('aer_simulator')
# qc_t = transpile(qc, backend)
# result = backend.run(qc_t).result()
# statevector = result.get_statevector(0)  # o result.data(0)['statevector']

# plot_bloch_multivector(statevector)
# plt.show()

############################################################

# from qiskit.visualization import plot_histogram

# qc = QuantumCircuit(1)
# initial_state = [0, 1] # Estado |1>
# qc.initialize(initial_state, 0)

# qc.x(0) # Aplicar una puerta X(NOT)
# qc.h(0) # Aplicar una puerta Hadamard(superposición)

# qc.save_statevector()  # <-- agregar esto
# backend = Aer.get_backend('aer_simulator')
# qc_t = transpile(qc, backend)
# result = backend.run(qc_t).result()
# statevector = result.get_statevector(0)  # o result.data(0)['statevector']

# # Visualizar histograma
# # plot_histogram(result.get_counts())
# # plt.show()

# # Visualizar el circuito
# qc.draw(output='mpl')
# plt.show()