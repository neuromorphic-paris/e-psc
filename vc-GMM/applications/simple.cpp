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
    if (argc < 11) {
        throw std::runtime_error(std::to_string(argc).append("received. Expected 11 arguments"));
    } else {
        
        // APPLICATION PARAMETERS
        std::string data_path           = argv[1];             // path to training features text file
        int C_p                         = std::atoi(argv[2]);  // number of clusters considered for each data point (1 <= C_p <= C)
        int G                           = std::atoi(argv[3]);  // search space (nearest neighbours for the C' clusters with 2 <= G <= C)
        bool plus1                      = std::atoi(argv[4]);  // include one additional randomly chosen cluster to each of the search spaces
        int N_core                      = std::atoi(argv[5]);  // size of subset
        int C                           = std::atoi(argv[6]);  // number of cluster centers
        int chain_length                = std::atoi(argv[7]);  // chain length for AFK-MC² seeding
        double convergence_threshold    = std::stod(argv[8]);  // convergence threshold
        bool save_centers               = std::atoi(argv[9]);  // write cluster centers to a text file
        bool save_prediction            = std::atoi(argv[10]); // write assigned clusters to a text file
        int seed                        = 123;                 // seed for random number generator
        int nthreads = std::thread::hardware_concurrency();    // number of C++11 threads
        
        // READING DATA
        blaze::DynamicMatrix<double, blaze::rowMajor> data_features;
        loadtxt(data_path, data_features); // read dataset
                
        // FITTING MODEL TO TRAINING DATASET
        clustering<double> vc_GMM(data_features, N_core, C, chain_length, C_p, G, plus1, nthreads, seed);
         
        vc_GMM.converge(convergence_threshold); // computes EM-iterations until the relative change in free energy falls below 1e-4
        auto cluster_assignments = vc_GMM.predict(); // returns the cluster assignments for the training data
        
        // writing the cluster centers to a text file
        if (save_centers) {
            savetxt("gmm_centers.txt", vc_GMM.cluster_centers());
        }
        
        if (save_prediction) {
            savevec("gmm_labels.txt", cluster_assignments);
        }
    }
    
    // EXIT PROGRAM
    return 0;
}
