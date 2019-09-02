/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#include "../source/clustering.hxx"

int main(void) {

    blaze::DynamicMatrix<double, blaze::rowMajor> x;
    loadtxt("<path-to-dataset>.txt", x); // read dataset from whitespace separated text file

    int C_p = 5;    // 1 <= C_p <= C
    int G   = 5;    // 2 <= G   <= C
    bool plus1      = true;

    int nthreads = std::thread::hardware_concurrency(); // number of C++11 threads
    int seed     = 123;                                 // seed for random number generator

    int N_core       = 10'000; // size of subset
    int C            = 200;    // number of cluster centers
    int chain_length = 20;     // chain length for AFK-MC² seeding

    clustering<double> vc_GMM(x, N_core, C, chain_length, C_p, G, plus1, nthreads, seed);

    vc_GMM.converge(1e-4); // computes EM-iterations until the relative change in free energy falls below 1e-4

    // uncomment to write the cluster centers to a text file, i.e. means of the GMM
    // savetxt("<path-to-result>.txt", vc_GMM.cluster_centers());

    std::cout << std::endl;
    std::cout << "quantization error: ";
    std::cout << vc_GMM.error(); // computes the quantization error
    std::cout << std::endl;
}
