from qiskit import QuantumCircuit, QuantumRegister, transpile, assemble, Aer, execute, IBMQ
from qiskit.circuit.library import EfficientSU2, QFT
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.ml.datasets import ad_hoc_data
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.ibmq import least_busy
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks, Optimize1qGates
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import ADAM
import random
from deap import creator, base, tools, algorithms
from qiskit.runtime import RuntimeEncoder, RuntimeDecoder
from qubovert import QAOA

IBMQ.save_account('YOUR_API_TOKEN', overwrite=True)  # Replace with your actual API token
provider = IBMQ.load_account()

# Try running on the least busy backend
backend_sim = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= feature_dim and not x.configuration().simulator and x.status().operational==True))
print("Running on backend:", backend_sim)

# Custom Feature Map and Variational Form Functions with Parameters
def custom_feature_map(x, params):
    n = len(x)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.u1(2 * np.pi * x[i] * params[i], i)  # Use the Parameter here
    qc.append(QFT(n, do_swaps=False), range(n))
    return qc

def custom_variational_form(n, depth, params):
    entangler_map = [[i, j] for i in range(n) for j in range(i + 1, n)]
    circuit = QuantumCircuit(n)

    for _ in range(depth):
        for i in range(n):
            circuit.rx(params[i], i)  # Use the Parameters here
            circuit.rz(params[i + n], i)

        for i, j in entangler_map:
            circuit.cz(i, j)

    return circuit

# Updating the predict function to handle multi-class classification
def multi_class_predict(vqc_classifier, data):
    probabilities = vqc_classifier.predict(datapoints[0])
    return np.argmax(probabilities, axis=1)

# Call the 'plot_decision_boundaries' function with the trained vqc and test input data
def plot_decision_boundaries(vqc_classifier, datapoints, class_to_label):
    h = .02  # step size in the mesh
    cmap_light = ListedColormap(['orange', 'cyan'])
    cmap_bold = ListedColormap(['darkorange', 'c'])
    xx, yy = np.meshgrid(np.arange(0, 2 * np.pi, h),
                         np.arange(0, 1 * np.pi, h))

    Z = np.array([multi_class_predict(vqc_classifier, np.array([xx, yy]).reshape(-1, 2))])
    Z = Z.reshape(xx.shape)

    plt.colorbar(plt.pcolormesh(xx, yy, Z, cmap=cmap_light))

    # Scatter plot of the data points
    scatter_x, scatter_y = datapoints[:, 0], datapoints[:, 1]

    for index in range(len(scatter_x)):
        plt.scatter(scatter_x[index], scatter_y[index], color=['red', 'blue'][int(datapoints[index][2])], marker='o')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature Dimension 1')
    plt.ylabel('Feature Dimension 2')
    plt.title('Variational Quantum Classifier Decision Boundaries')
    plt.show()

# Customizing transpilation to improve the computational efficiency
def custom_transpile(circuit):
    pm = PassManager()
    pm.append([Layout.from_qubit_dict({qr[0]: 2, qr[1]: 1})])
    pm.append(Collect2qBlocks())
    pm.append(ConsolidateBlocks(basis_gates=['id', 'u1', 'u2', 'u3', 'cx']))
    pm.append(Optimize1qGates())
    circuit = pm.run(circuit)
    return circuit

# Prepare dataset
feature_dim = 2
training_dataset_size = 20
testing_dataset_size = 10
random_seed = 10598
shots = 1024

sample_Total, training_input, test_input, class_labels = ad_hoc_data(
    training_size=training_dataset_size,
    test_size=testing_dataset_size,
    n=feature_dim,
    gap=0.3,
    plot_data=False
)

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)

# Define eval_fitness function
def eval_fitness(individual):
    best_feature_map_params = individual[:feature_dim]
    best_var_form_params = individual[feature_dim:]

    backend_sim = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= feature_dim and not x.configuration().simulator and x.status().operational == True))

    # Create the quantum instance with noise mitigation
    qr = QuantumRegister(feature_dim)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    cal_results = execute(meas_calibs, backend=backend_sim, shots=shots, optimization_level=0)
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    quantum_instance = QuantumInstance(backend_sim, shots=shots,
                                       measurement_error_mitigation_cls=CompleteMeasFitter,
                                       cals_matrix_refresh_period=30,
                                       seeding_strategy='random',
                                       measurement_error_mitigation_shots=shots,
                                       seed_simulator=random_seed,
                                       seed_transpiler=random_seed)

    # Create the feature map and variational form
    feature_map = lambda x: custom_feature_map(x, best_feature_map_params)
    var_form = lambda: custom_variational_form(feature_dim, depth=2, params=best_var_form_params)

    # Create the variational quantum classifier
    vqc = VQC(optimizer=COBYLA(maxiter=500, tol=0.001),
              feature_map=feature_map,
              var_form=var_form,
              training_dataset=training_input,
              test_dataset=test_input)

    # Run the VQC and retrieve the result
    result = vqc.run(quantum_instance,
                     transpile_callback=custom_transpile,
                     batch_mode=True)

    # Calculate the fitness value (minimize error)
    fitness = 1.0 - result['testing_accuracy']

    return (fitness,)

# Set up the Genetic Algorithm
population_size = 50
num_generations = 10

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 2 * np.pi)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2 * feature_dim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxUniform, indpb=0.1)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=2 * np.pi, indpb=0.1)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", eval_fitness)

pop = toolbox.population(n=population_size)
hof = tools.HallOfFame(1)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof)

# Get the best parameters
best_params = hof[0]
best_feature_map_params = best_params[:feature_dim]
best_var_form_params = best_params[feature_dim:]

# Train the VQC with the best parameters
vqc = VQC(optimizer=COBYLA(maxiter=500, tol=0.001),
          feature_map=lambda x: custom_feature_map(x, best_feature_map_params),
          var_form=lambda: custom_variational_form(feature_dim, depth=2, params=best_var_form_params),
          training_dataset=training_input,
          test_dataset=test_input)

# Setting up noise mitigation
qr = QuantumRegister(feature_dim)
meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
cal_results = execute(meas_calibs, backend=backend_sim, shots=shots, optimization_level=0)
meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
quantum_instance = QuantumInstance(backend_sim, shots=shots,
                                   measurement_error_mitigation_cls=CompleteMeasFitter,
                                   cals_matrix_refresh_period=30,
                                   seeding_strategy='random',
                                   measurement_error_mitigation_shots=shots,
                                   seed_simulator=random_seed,
                                   seed_transpiler=random_seed)

result = vqc.run(quantum_instance,
                 transpile_callback=custom_transpile,         # Adding the custom transpile option
                 batch_mode=True)                             # Set batch mode to true

# Print the final testing success ratio
print(f"Testing success ratio: {result['testing_accuracy']}")

# Call the 'plot_decision_boundaries' function with the trained VQC and test input data
plot_decision_boundaries(vqc, datapoints, class_to_label)
