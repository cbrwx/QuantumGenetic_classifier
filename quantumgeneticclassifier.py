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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import ADAM
import random
from deap import creator, base, tools, algorithms

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

def plot_decision_boundaries(vqc_classifier, datapoints, class_to_label):
    h = .02  # step size in the mesh
    cmap_light = ListedColormap(['orange', 'cyan'])
    cmap_bold = ListedColormap(['darkorange', 'c'])
    xx, yy = np.meshgrid(np.arange(0, 2 * np.pi, h),
                         np.arange(0, 1 * np.pi, h))

    Z = np.array([vqc_classifier.predict(np.array([xx, yy]).reshape(-1, 2))])
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

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

# Using the Genetic Algorithm to optimize feature map and variational form parameters
def eval_circuit(params):
    feature_map_params = params[:feature_dim]
    var_form_params = params[feature_dim:]

    vqc = VQC(optimizer=COBYLA(maxiter=1, tol=0.001),
              feature_map=lambda x: custom_feature_map(x, feature_map_params),
              var_form=lambda: custom_variational_form(feature_dim, depth=2, params=var_form_params),
              training_dataset=training_input,
              test_dataset=test_input)

    quantum_instance = QuantumInstance(backend_sim, shots=shots, seed_simulator=random_seed)
    result = vqc.run(quantum_instance)
    return (1 - result['testing_accuracy'],)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("uniform_random", random.uniform, 0, 2 * np.pi)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.uniform_random, 2 * feature_dim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=np.pi / 8, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_circuit)

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

# Set up the Genetic Algorithm
population_size = 50
num_generations = 10
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

quantum_instance = QuantumInstance(backend_sim, shots=shots, seed_simulator=random_seed, seed_transpiler=random_seed)
result = vqc.run(quantum_instance)

# Print the final testing success ratio
print(f"Testing success ratio: {result['testing_accuracy']}")

# Call the 'plot_decision_boundaries' function with the trained VQC and test input data
plot_decision_boundaries(vqc, datapoints, class_to_label)
