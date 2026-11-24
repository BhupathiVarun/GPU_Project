#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

using namespace std;

int main(int argc, char* argv[])
{
    size_t N = 256;
    const int RUNS = 5;

    

    // Read matrix size
    if (argc > 1) {
        char* end = nullptr;
        unsigned long long value = strtoull(argv[1], &end, 10);

        if (end == argv[1] || *end != '\0' || value == 0) {
            cerr << "Invalid matrix size!" << endl;
            return 1;
        }

        N = static_cast<size_t>(value);
    }

    // Check for size overflow
    if (N > numeric_limits<size_t>::max() / N) {
        cerr << "Matrix size is too large!" << endl;
        return 1;
    }

    size_t total = N * N;

    try {
        vector<double> A(total);
        vector<double> B(total);
        vector<double> C(total, 0.0);

        // Initialize deterministic, non-constant input matrices
        for (size_t i = 0; i < total; i++) {
            A[i] = (static_cast<int>((i * 17 + 3) % 101) - 50) / 100.0;
            B[i] = (static_cast<int>((i * 13 + 7) % 101) - 50) / 100.0;
        }

        vector<double> times;

        // One warm-up run, followed by five measured runs
        for (int run = 0; run <= RUNS; run++) {

            fill(C.begin(), C.end(), 0.0);

            auto start = chrono::steady_clock::now();

            // Baseline matrix multiplication: i-j-k
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    for (size_t k = 0; k < N; k++) {
                        C[i * N + j] +=
                            A[i * N + k] * B[k * N + j];
                    }
                }
            }

            auto end = chrono::steady_clock::now();

            chrono::duration<double> elapsed = end - start;

            if (run > 0) {
                times.push_back(elapsed.count());
            }
        }

        // Calculate average execution time
        double average = 0.0;

        for (double t : times) {
            average += t;
        }

        average /= RUNS;

        // Calculate median execution time
        vector<double> sorted_times = times;
        sort(sorted_times.begin(), sorted_times.end());

        double median = sorted_times[RUNS / 2];

        // Calculate GFLOP/s using median execution time
        double operations = 2.0 * static_cast<double>(N)
                          * static_cast<double>(N)
                          * static_cast<double>(N);

        double gflops = operations / median / 1e9;

        // Numerical correctness check using independent reference calculation
        bool correct = true;
        double max_error = 0.0;

        for (size_t i = 0; i < N && correct; i++) {
            for (size_t j = 0; j < N; j++) {

                double reference = 0.0;

                for (size_t k = 0; k < N; k++) {
                    reference += A[i * N + k] * B[k * N + j];
                }

                double error = abs(C[i * N + j] - reference);

                max_error = max(max_error, error);

                double tolerance = 1e-9 + 1e-9 * abs(reference);

                if (error > tolerance) {
                    correct = false;
                    break;
                }
            }
        }

        cout << fixed << setprecision(6);

        cout << "\n===== CPU BASELINE MATRIX MULTIPLICATION =====\n";
        cout << "Matrix size       : " << N << " x " << N << endl;
        cout << "Loop order        : i-j-k" << endl;
        cout << "Warm-up runs      : 1" << endl;
        cout << "Measured runs     : " << RUNS << endl;
        cout << "Average time      : " << average << " seconds\n";
        cout << "Median time       : " << median << " seconds\n";
        cout << "Performance       : " << gflops << " GFLOP/s\n";
        cout << "C[0][0]           : " << C[0] << endl;
        cout << "C[N-1][N-1]       : " << C[total - 1] << endl;
        cout << "Maximum error     : " << max_error << endl;
        cout << "Correctness       : "
             << (correct ? "PASSED" : "FAILED") << endl;

        return correct ? 0 : 1;
    }
    catch (const bad_alloc&) {
        cerr << "Memory allocation failed. Try a smaller matrix size."
             << endl;
        return 1;
    }
}

