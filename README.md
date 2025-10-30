SP23-BCS-096 MidLab

This repository contains all the tasks, codes, outputs, and analysis for the Parallel and Distributed Computing MidLab. The lab focuses on text analytics on the IMDB dataset using Sequential, Multiprocessing, MPI, and Hybrid approaches.

Contents
Task	Description	File(s)	Output
Task 1	Sequential Text Analysis	task01.py	seq_output.csv
Task 2	Multiprocessing Text Analysis	task02.py	para_output.csv
Task 3	MPI Text Analysis	task03.py	mpi_top_words.csv
Task 4	Hybrid MPI + Multiprocessing Analysis	task04.py	hybrid_output.csv
Project Overview

Objective:
Analyze the frequency of words in IMDB reviews using different parallel and distributed computing methods.

Dataset:

IMDB Dataset.csv (contains movie reviews and sentiments)

Subset of 20,000–50,000 reviews is used for performance analysis.

Methods Implemented:

Sequential: Basic Python processing with NLTK stopwords.

Multiprocessing: Python multiprocessing with 1, 2, 4, 8 workers.

MPI: Distributed computation using mpi4py.

Hybrid: Combination of MPI across nodes and multiprocessing threads per node.

Performance Analysis:

Sequential baseline: 4.1s

Multiprocessing (2 workers) achieved the best speedup: 2.15x

MPI and Hybrid methods underperformed due to communication and overhead costs.

Graph of Speedup vs Number of Cores is included.

Usage Instructions

Clone the repository:

git clone https://github.com/<your-username>/sp23-bcs-096-midlab.git
cd sp23-bcs-096-midlab


Install required packages:

pip install pandas nltk mpi4py matplotlib


Run any task:

python task01.py    # Sequential
python task02.py    # Multiprocessing
mpiexec -n 4 python task03.py   # MPI
mpiexec -n 2 python task04.py   # Hybrid


Outputs:

seq_output.csv, para_output.csv, mpi_top_words.csv, hybrid_output.csv

Conclusion

Best approach for small to medium datasets (20K–50K): Multiprocessing with 2–4 workers.

Distributed computing (MPI/Hybrid) is more suitable for large datasets (>500K reviews).

Choosing the right method depends on dataset size, system resources, and parallelization overhead.
