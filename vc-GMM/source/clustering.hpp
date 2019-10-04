/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
//  Modified by Omar Oubari <omar.oubari@inserm.fr>
//  Added Prediction method
/******************************************************************************/

#include "variational.hpp"
#include "subsample.hpp"
#include "seeding.hpp"
#include "threads.hpp"
#include "utility.hpp"

template <typename T>
struct clustering {
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x; // dataset
    blaze::DynamicMatrix<T, blaze::rowMajor> c;        // coreset
    blaze::DynamicVector<T> w;                         // weights
    blaze::DynamicMatrix<T, blaze::rowMajor> s;        // cluster center
    variational<T> algo;                               // implements var-GMM-S and vc-GMM
    bool initial;

public:
    // wrapper for var-GMM-S
    clustering( const blaze::DynamicMatrix<T, blaze::rowMajor>& x, std::size_t C, std::size_t chain, std::size_t C_p, std::size_t G, bool plus1, std::size_t nthreads, std::size_t seed) :
            x(x),
            w(x.rows(), T(1.0)),
            s(C, x.columns()),
            algo(x, w, s, C_p, G, plus1, 1.0, nthreads, seed),
            initial(true) {
        afkmc2(x, w, s, algo.mt[0][0], C, chain);   // AFK-MC^2 seeding
    }

    // wrapper for vc-GMM
    clustering(const blaze::DynamicMatrix<T, blaze::rowMajor>& x, std::size_t N_core, std::size_t C, std::size_t chain, std::size_t C_p, std::size_t G, bool plus1, std::size_t nthreads, std::size_t seed) :
            x(x),
            c(N_core, x.columns()),
            w(N_core),
            s(C, x.columns()),
            algo(c, w, s, C_p, G, plus1, 1.0, nthreads, seed),
            initial(true) {
        lwcs(x, c, w, N_core, algo.mt[0][0]);       // construct lightweight coreset
        afkmc2(c, w, s, algo.mt[0][0], C, chain);   // AFK-MC^2 seeding
    }

    void converge(T eps) {

        double prv = 0.0;
        double cur = 0.0;
        size_t iters = 0;

        while (true) {

            // E-STEP
            algo.estep();
            
            // M-STEP
            if (initial) {
                initial = false;
                algo.guess_variance();
                algo.fit();
            } else {
                algo.fit();
            }
            iters++;

            cur = algo.free_energy;

            /* decrease in free energy */
            std::cout << std::scientific;
            std::cout << "iteration "   << iters            << "\t";
            std::cout << "free energy " << algo.free_energy << "\t";
            std::cout << "variance "    << algo.variance    << "\n";
            std::cout << std::flush;

            if (iters > 1) {

                /* convergence criterion */
                if ((std::abs((cur - prv) / cur) < eps)) {
                    break;
                }
            }
            prv = cur;
        }
    }
    
    // returns the cluster assignments for the training data
    std::vector<size_t> predict() {
        return inference(x, s, algo.threads);
    }
    
    // predicts the clusters of the given testing data
    std::vector<size_t> predict(const blaze::DynamicMatrix<T, blaze::rowMajor>& data) {
        return inference(data, s, algo.threads);
    }
    
    blaze::DynamicMatrix<T, blaze::rowMajor> cluster_centers(void) {
        return s;
    }

    void set_variance(double variance) {
        algo.variance = variance;
        initial = false;
    }

    T variance(void) {
        return algo.variance;
    }

    T lower_bound(void) {
        return algo.free_energy;
    }

    T error(void) {
        return quantization(x, s, algo.threads);
    }
};
