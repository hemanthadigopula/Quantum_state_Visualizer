import streamlit as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization.bloch import Bloch
import matplotlib.pyplot as plt
import numpy as np

# =====================
# Utility functions
# =====================

def strip_nonunitary(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of the circuit with only unitary gates (skip measure/reset/barrier/classical)."""
    new_circ = QuantumCircuit(circuit.num_qubits)
    for instr, qargs, cargs in circuit.data:
        if instr.name in ["measure", "reset", "barrier"]:
            continue
        if len(cargs) > 0:  # skip classical-controlled ops
            continue
        new_circ.append(instr, qargs)
    return new_circ


def get_single_qubit_rdm(circuit: QuantumCircuit):
    """Return reduced density matrices (RDMs) for each qubit of the circuit."""
    circuit = strip_nonunitary(circuit)
    state = Statevector.from_instruction(circuit)

    rdm_list = []
    for i in range(circuit.num_qubits):
        traced = partial_trace(state, [j for j in range(circuit.num_qubits) if j != i])
        rdm_list.append(traced)
    return rdm_list


def bloch_vector_from_rdm(rdm):
    """Convert a single-qubit density matrix into a Bloch vector [x,y,z]."""
    rho = np.array(rdm.data, dtype=complex)
    return [
        2 * np.real(rho[0, 1]),            # X expectation
        2 * np.imag(rho[1, 0]),            # Y expectation
        np.real(rho[0, 0] - rho[1, 1])     # Z expectation
    ]


def plot_all_qubits(circuit: QuantumCircuit):
    """Render a Bloch sphere with vector for each qubit separately."""
    rdm_list = get_single_qubit_rdm(circuit)

    for i, rdm in enumerate(rdm_list):
        bloch_vec = bloch_vector_from_rdm(rdm)

        b = Bloch()
        b.add_vectors(bloch_vec)
        b.title = f"Qubit {i}"
        b.render()               # <-- no arguments here
        st.pyplot(b.fig)         # <-- directly use the Bloch's own figure


# =====================
# Streamlit App UI
# =====================

st.set_page_config(page_title="Quantum State Visualizer", page_icon="⚛️", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: black; color: #00BFFF; }
    .stTextArea textarea { background-color: #001F3F; color: #00BFFF; font-size: 14px; }
    .stButton button { background-color: #00BFFF; color: black; font-weight: bold; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚛️ Quantum Circuit Bloch Sphere Visualizer")
st.write("Enter a QASM code to visualize qubit states with **vector arrows** on Bloch spheres.")

# Text input for QASM
qasm_input = st.text_area("Paste your QASM code here:", height=200)

if st.button("Visualize"):
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_input)
        st.write("### Quantum Circuit:")
        st.pyplot(qc.draw("mpl"))

        st.write("### Single-Qubit States on Bloch Spheres:")
        plot_all_qubits(qc)

    except Exception as e:
        st.error(f"Error: {str(e)}")
