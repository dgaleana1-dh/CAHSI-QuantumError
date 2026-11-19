import warnings

# Silence Qiskit transpiler UserWarnings and general warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
import os
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
# Import Tim's compare_fez
import compare_fez


GATE_SET = "x"
MIRROR = False

def run_experiment(circuit_type, num_gates, shots, seed, backend, ideal_sim,layouts):
    base_qc = compare_fez.test_circuit(
        gate_set=GATE_SET, num_gates=num_gates, seed=seed, mirror=MIRROR
    )

    plain = base_qc.copy()
    plain.measure_all()

    if circuit_type == "unencoded":
        circuits, _ = compare_fez.on_fez(plain, {}, backend, layouts, seed)
        t_noisy = circuits[f"no correction (Fez {layouts['bit/phase-flip'][0]})"]
    else:
        encoded = {circuit_type: compare_fez.CODES[circuit_type](base_qc)}
        circuits, _ = compare_fez.on_fez(plain, encoded, backend, layouts, seed)
        t_noisy = circuits[circuit_type]

    # 1. Ideal run
    ideal_counts = (
        ideal_sim.run(plain, shots=shots, seed_simulator=seed)
        .result()
        .get_counts()
    )

    # 2. Noisy run
    noisy_sim = AerSimulator.from_backend(backend)
    noisy_counts = (
        noisy_sim.run(t_noisy, shots=shots, seed_simulator=seed)
        .result()
        .get_counts()
    )

    # Extract output bit probabilities using Tim's syndrome parser
    ones_ideal, _ = compare_fez.prob_one(ideal_counts)
    ideal_p1 = ones_ideal / shots
    ideal_p0 = 1.0 - ideal_p1

    ones_noisy, _ = compare_fez.prob_one(noisy_counts)
    observed_p1 = ones_noisy / shots
    observed_p0 = 1.0 - observed_p1

    # Tim's output-probability difference error metric
    error_metric = abs(observed_p1 - ideal_p1)

    return {
        "circuit_type": circuit_type,
        "num_gates": num_gates,
        "shots": shots,
        "random_seed": seed,
        "noise_model": "FakeFez",
        "ideal_p0": f"{ideal_p0:.4f}",
        "ideal_p1": f"{ideal_p1:.4f}",
        "observed_p0": f"{observed_p0:.4f}",
        "observed_p1": f"{observed_p1:.4f}",
        "error_metric": f"{error_metric:.4f}",
    }


def main():
    csv_filename = "csvs/benchmarking_experiments.csv"
    headers = [
        "circuit_type",
        "num_gates",
        "shots",
        "random_seed",
        "noise_model",
        "ideal_p0",
        "ideal_p1",
        "observed_p0",
        "observed_p1",
        "error_metric",
    ]

    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    backend = FakeFez()
    ideal_sim = AerSimulator()
    layouts = compare_fez.set_layout(backend)

    # Tested circuit types matching compare_fez.CODES keys
    circuit_types = ["unencoded"] + [
        k for k in compare_fez.CODES.keys() if "if blocks" in k
    ]
    gate_counts = [10, 50, 100, 500]  # different circuit lengths
    shot_counts = [10000]  # 10,000 simulations
    seeds = [3, 50, 100]  # (At least 3 seeds)

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for c_type in circuit_types:
            for gates in gate_counts:
                for shots in shot_counts:
                    for seed in seeds:
                        res = run_experiment(
                            c_type, gates, shots, seed, backend, ideal_sim, layouts
                        )
                        writer.writerow(res)
                        print(f"Recorded: {res['circuit_type']:<22} | Gates: {res['num_gates']:<3} | Seed: {res['random_seed']:<3} | Error: {res['error_metric']}")

    print(f"\n Benchmark completed. All results saved to '{csv_filename}'.")


if __name__ == "__main__":
    main()