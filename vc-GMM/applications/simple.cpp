/******************************************************************************/
// simple.cpp
//
// Created by Omar Oubari
// Email: omar.oubari@inserm.fr
//
// Information: simple clustering test to verify the algorithm
/******************************************************************************/

#include <string>
#include "../source/clustering.hpp"

int main(int argc, char** argv) {
    // APPLICATION PARAMETERS
    std::string data_path = "../datasets/simple/data.txt";
    int C_p                         = 5;                // number of clusters considered for each data point (1 <= C_p <= C)
    int G                           = 5;                // search space (nearest neighbours for the C' clusters with 2 <= G <= C)
    bool plus1                      = true;             // include one additional randomly chosen cluster to each of the search spaces
    int N_core                      = 1000;             // size of subset
    int C                           = 15;               // number of cluster centers
    int chain_length                = 20;               // chain length for AFK-MC² seeding
    double convergence_threshold    = 0.0001;           // convergence threshold
    bool save                       = true;             // write cluster centers to a text file
    int seed                        = 123;              // seed for random number generator
    int nthreads = std::thread::hardware_concurrency(); // number of C++11 threads
    
    // READING DATA
    blaze::DynamicMatrix<double, blaze::rowMajor> train_features;
    loadtxt(path_train_features, train_features); // read dataset
    
    // FITTING MODEL TO TRAINING DATASET
    clustering<double> vc_GMM(train_features, N_core, C, chain_length, C_p, G, plus1, nthreads, seed);

    vc_GMM.converge(convergence_threshold); // computes EM-iterations until the relative change in free energy falls below 1e-4
    
    // writing the cluster centers to a text file
    if (save) {
        savetxt("results/cluster_centers.txt", vc_GMM.cluster_centers());
    }
    
    // EXIT PROGRAM
    return 0;
}
