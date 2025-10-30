import matplotlib.pyplot as plt

# Data
methods = ['Multiprocessing', 'MPI', 'Hybrid']
cores = {
    'Multiprocessing': [1, 2, 4, 8],
    'MPI': [1],
    'Hybrid': [4]  # threads per process shown as "cores"
}
speedup = {
    'Multiprocessing': [1.0, 1.44, 1.50, 1.13],
    'MPI': [1.0],
    'Hybrid': [0.62]
}

# Plot
plt.figure(figsize=(8,5))
for method in methods:
    plt.plot(cores[method], speedup[method], marker='o', label=method)

plt.xlabel('Number of Cores / Workers')
plt.ylabel('Speedup')
plt.title('Speedup vs. Number of Cores / Workers')
plt.grid(True)
plt.xticks([1,2,4,8])
plt.legend()
plt.show()
