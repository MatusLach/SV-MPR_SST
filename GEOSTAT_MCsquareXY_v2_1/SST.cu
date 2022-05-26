/* SST.cu

GEOSTATISTICAL APPLICATION "v2_1" uses checkerboard spin-flip Metropolis simulation
of a two-dimensional ferromagnetic XY model with modified hamiltonian
H = -J sum_{ij} cos( Qfactor*(theta_i - theta_j) )
on graphics processing units (GPUs) using the NVIDIA CUDA framework.

Implements spatially variable MPR using single checkerboard decomposition and site-specific temperatures.
*/

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif


#include <iostream>
#include <fstream>
//#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstdio>
#define _USE_MATH_DEFINES	// for pi constant
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <device_functions.h> internal header, deprecated in the future, misov povodny kod
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <vector>
//#include <list>
#include <iterator>
#include <algorithm> 		// for sort operation
#include <limits>			// has own min(), max() ???? which behave correctly with NAN (ignoing NAN values)
#include <windows.h>
#include <random>

#include <cub/cub.cuh>

#undef min
#undef max

#include <chrono>	// high precision execution time measurment

/*#include <thread>*/
//#include <boost/timer/timer.hpp>

//using namespace std;

#define DIM 2

#define L 256    // minumum of L = 2*BLOCKL

#define Qfactor 0.5f	// uncomment for single precision calculations in kernel
#define BLOCKL 32
#define GRIDL (L/BLOCKL)
#define BLOCKS ((GRIDL*GRIDL)/2)
#define THREADS ((BLOCKL*BLOCKL)/2)
#define N (L*L)
#define Nbond (2*L*(L - 1))
//#define TOTTHREADS (BLOCKS*THREADS)



#define SWEEPS_GLOBAL 100
#define SWEEPS_EMPTY 1
#define CONFIG_SAMPLES 100		// M = 100


#define ACC_RATE_MIN_EQ 0.30		// A_targ = 0.3
#define ACC_RATE_MIN_SIM 0.30		


//#define ACC_TEST_FREQUENCY_EQ 10	//ACC_TEST_FREQUENCY_EQ 
//#define ACC_TEST_FREQUENCY_SIM 10	//ACC_TEST_FREQUENCY_SIM
#define EQUI_TEST_FREQUENCY 5   // n_f = 5
#define EQUI_TEST_SAMPLES 20    // n_fit povodne 20



#define SWEEPS_EQUI_MAX 300		// upper limit for equilibration hybrid sweeps; probably not necessary
#define SLOPE_RESTR_FACTOR 3.0	// k_a = 3; for a = 1 + i/k_a (SLOPE_RESTR = k_a)
#define SMOOTHING_PARAMETER 0  //number of times the smoothing procedure is carried out

#define RemovedDataRatio 0.8f


//#define SOURCE_DATA_PATH "zo_L2048_ka02_nu05.bin"
#define SOURCE_DATA_PATH "walker_lake.bin"
//#define SOURCE_DATA_PATH "wall_3_L2048.bin"
#define SOURCE_DATA_NAME "walker"
/*#define RNG_SEED_DILUTION 1564564ULL
#define RNG_SEED_FILL 25756655ULL
#define RNG_SEED_SIMULATION 3456454624ULL*/
#define RNG_SEED_DILUTION 842301111UL
#define RNG_SEED_FILL 5451UL
#define RNG_SEED_SIMULATION 3645445443UL

//#define DOUBLE_PRECISION
#ifndef DOUBLE_PRECISION
#define INTRINSIC_FLOAT
#endif

#define OVER_RELAXATION_EQ
//#define OVER_RELAXATION_SIM

//comment these for time measurements
#define ENERGIES_PRINT
//#define CONFIGURATION_PRINT
//#define RECONSTRUCTION_PRINT
//#define ERROR_PRINT
//#define TEMPSnALPHAS_PRINT
//#define RANDOM_INIT
//#define DIL_ENERGIES_PRINT	// not working yet

//#define SOURCE_MAPPING

//#define COLD_START			// not working yet

// other macros
// linear congruential generator
#define AA 1664525
#define CC 1013904223
#define RAN(n) (n = AA*n + CC)
#define MULT 2.328306437080797e-10f
/*
#define MULT2 4.6566128752457969e-10f
*/
#define sS(x,y) sS[(y+1)*(BLOCKL+2)+x+1]

typedef double source_t;
#ifdef DOUBLE_PRECISION
typedef double spin_t;
typedef double energy_t;
#else
typedef float spin_t;
typedef float energy_t;
#endif

// GPU processing partition
const dim3 gridLinearLattice((int)ceil(N / 256.0));
const dim3 gridLinearLatticeHalf((int)ceil(N / 2.0 / 256.0));
const dim3 blockLinearLattice(256);

//pre double checkerboard
const dim3 grid(GRIDL, GRIDL / 2);
const dim3 block(BLOCKL, BLOCKL / 2);
//pre single checkerboard
const dim3 grid_check(GRIDL, GRIDL);
const dim3 block_check(BLOCKL, BLOCKL / 2);

//hotfixed * 2
const dim3 gridAcc((int)ceil(BLOCKS * 2 / 128.0));
const dim3 blockAcc(128);

const dim3 gridEn(GRIDL, GRIDL);
const dim3 blockEn(BLOCKL, BLOCKL);


// CUDA error checking macro
#define CUDAErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s ; %s ; line %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// cuRAND error checking macro
#define cuRAND_ErrChk(err) { if (err != CURAND_STATUS_SUCCESS) std::cout << curandGetErrorString(err) << "\n"; }

// cuRAND errors
char* curandGetErrorString(curandStatus_t);
const char* curanderr[14] = {
    "No errors", "Header file and linked library version do not match",
    "Generator not initialized", "Memory allocation failed",
    "Generator is wrong type", "Argument out of range",
    "Length requested is not a multiple of dimension",
    "GPU does not have double precision required by MRG32k3a",
    "Kernel launch failure", "Pre-existing failure on library entry",
    "Initialization of CUDA failed", "Architecture mismatch, GPU does not support requested feature",
    "Internal library error", "Unknown error"
};


//single checkerboard
__global__ void metro_conditioned_sublattice_k(spin_t*, spin_t*, float*, unsigned int, energy_t*, unsigned int*, spin_t*);
__global__ void spin_mult(spin_t*, spin_t);
__global__ void over_relaxation_k(spin_t*, spin_t*, int);
__global__ void energyCalc_k(spin_t*, energy_t*);
__global__ void energyCalcDiluted_k(spin_t*, energy_t*);
__global__ void energyCalcDiluted_per_block(energy_t*, unsigned int*, unsigned int);
__global__ void min_max_avg_block(spin_t*, spin_t*, spin_t*, spin_t*);
__global__ void resetAccD_k(energy_t*);
__global__ void min_max_k(source_t*, source_t*, source_t*, bool, spin_t*);
__global__ void XY_mapping_k(source_t*, spin_t*, source_t, source_t, bool, spin_t*);
__global__ void create_dilution_mask_k(spin_t*, float*, unsigned int*);
__global__ void fill_lattice_nans_random(spin_t*, spin_t*);
__global__ void fill_lattice_nans_averaged(spin_t*, float*);
__global__ void fill_lattice_nans_averaged_global(spin_t*, spin_t);
__global__ void data_reconstruction_k(source_t*, spin_t*, source_t, source_t, source_t*, source_t*);
__global__ void mean_stdDev_reconstructed_k(source_t*, source_t*, unsigned int);
__global__ void sum_prediction_errors_k(source_t*, source_t*, spin_t*, source_t*, source_t*, source_t*, source_t*, source_t*, source_t*);
__global__ void sum_prediction_errors_k(source_t*, source_t*, spin_t*, source_t*, source_t*, source_t*, source_t*);
__global__ void bondCount_k(spin_t*, unsigned int*);
__global__ void find_temperature_gpu(energy_t*, double*, double*, energy_t*, int, int);
__global__ void correctTemps(energy_t*, energy_t);
__global__ void smoothTemps(energy_t*, energy_t*, energy_t*, bool);
__global__ void setInitialLocalAlphas(spin_t*, spin_t*);
__global__ void setLocalAlphas(unsigned int*, spin_t*, int, energy_t);
std::vector<energy_t> smoothTempsCPU(energy_t*);
energy_t cpu_energy(spin_t*);
double find_temperature(energy_t, std::vector<double>, std::vector<double>);

template <class T> T sumPartialSums(T *, int);
template <class T> std::vector<T> findMinMax(T *, T *, int);
template <class T> T find_median(T *, int);
template <int BLOCK_THREADS, int ITEMS_PER_THREAD> __global__ void BlockSortKernel(energy_t *d_in, energy_t *d_out);


int main()
{

    std::cout << "---Single checkerboard algorithm with local smooth temperatures---\n"

        << "\nRECONSTRUCTION SIMULATION CONFIGURATION:\n"
        << "L = " << L << ",\tQfactor = " << Qfactor << "\n"
        << "BLOCKL = " << BLOCKL << "\n"
        << "Missing data = " << RemovedDataRatio * 100 << "%\n"
        << "Equilibration samples for convergence testing = " << EQUI_TEST_SAMPLES << "\n"
        << "Reconstruction samples = " << SWEEPS_GLOBAL << "\n"
        << "Configuration samples = " << CONFIG_SAMPLES << "\n" << "Active macros: ";
#ifdef DOUBLE_PRECISION
    std::cout << " DOUBLE_PRECISION,";
#else
    std::cout << " SINGLE_PRECISION,";
#ifdef INTRINSIC_FLOAT
    std::cout << " INTRINSIC_FLOAT,";
#endif
#endif
#ifdef ENERGIES_PRINT
    std::cout << " ENERGIES_PRINT,";
#endif
#ifdef CONFIGURATION_PRINT
    std::cout << " CONFIGURATION_PRINT,";
#endif
#ifdef RECONSTRUCTION_PRINT
    std::cout << " RECONSTRUCTION_PRINT,";
#endif
#ifdef ERROR_PRINT
    std::cout << " ERROR_PRINT,";
#endif
#ifdef TEMPSnALPHAS_PRINT
    std::cout << " TEMPSnALPHAS_PRINT,";
#endif 

    
#ifdef OVER_RELAXATION_EQ
    std::cout << " OVER_RELAXATION_EQ,";
#endif
#ifdef OVER_RELAXATION_SIM
    std::cout << " OVER_RELAXATION_SIM,";
#endif
#ifdef SOURCE_MAPPING
    std::cout << " SOURCE_MAPPING,";
#endif
#ifdef RANDOM_INIT
    std::cout << " RANDOM_INIT,";
#endif
    std::cout << "\n";

    // time measurement - entire process
    std::chrono::high_resolution_clock::time_point t_sim_begin = std::chrono::high_resolution_clock::now();

    /* time measurement - relevant parts for geostatistical calulation
    (loading reference E = E(T), loading source, mapping to XY model, equilibration and reconstruction sample collection)
    */
    std::chrono::high_resolution_clock::time_point t_geo_begin;
    std::chrono::high_resolution_clock::time_point t_geo_end;

    t_geo_begin = std::chrono::high_resolution_clock::now();

    //std::cout << "------ LOADING REFERENCES AND SOURCE DATA ------\n";

    // read reference energies and temperatures
    char *buffer;
    const int ref_size = 1100;
    std::vector<double> T_ref;
    std::ifstream fileT("./reference/reference_T.bin", std::ios::in | std::ios::binary);
    buffer = (char*)malloc(ref_size * sizeof(double));
    fileT.read(buffer, ref_size * sizeof(double));
    T_ref.assign(reinterpret_cast<double*>(buffer), reinterpret_cast<double*>(buffer) + ref_size);
    fileT.close();

    std::vector<double> E_ref;
    std::ifstream fileE("./reference/reference_E.bin", std::ios::in | std::ios::binary);
    fileE.read(buffer, ref_size * sizeof(double));
    E_ref.assign(reinterpret_cast<double*>(buffer), reinterpret_cast<double*>(buffer) + ref_size);
    fileE.close();

    free(buffer);

    //allocate memory and copy reference energies and temperatures to the GPU
    double *T_ref_d, *E_ref_d;
    CUDAErrChk(cudaMalloc((void**)&T_ref_d, ref_size * sizeof(double)));
    CUDAErrChk(cudaMalloc((void**)&E_ref_d, ref_size * sizeof(double)));
    CUDAErrChk(cudaMemcpy(T_ref_d, T_ref.data(), ref_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDAErrChk(cudaMemcpy(E_ref_d, E_ref.data(), ref_size * sizeof(double), cudaMemcpyHostToDevice));

    
    // read data source
#ifdef SOURCE_DATA_PATH
    std::cout << "Source data: " << SOURCE_DATA_PATH << "\n";
    std::ifstream fileSource(SOURCE_DATA_PATH, std::ios::in | std::ios::binary);
    std::vector<source_t> complete_source;
    buffer = (char*)malloc(N * sizeof(source_t));
    fileSource.read(buffer, N * sizeof(source_t));
    complete_source.assign(reinterpret_cast<source_t*>(buffer), reinterpret_cast<source_t*>(buffer) + N);
    fileSource.close();
    free(buffer);
#else
    std::cout << "Source data path not specified!";
    return 0;
#endif
    std::cout << "Source size: " << complete_source.size() << "\n";


    //cudaSetDevice(0);


    // allocate GPU memory for source data, mapped data (XY model) and dilution mask (array of ones and NANs) & other variables
    source_t *source_d, *reconstructed_d, *mean_recons_d, *stdDev_recons_d, *AAE_d, *ARE_d, *AARE_d, *RASE_d;
#ifdef ERROR_PRINT
    source_t *error_map_d, *error_map_block_d;
    CUDAErrChk(cudaMalloc((void**)&error_map_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMemset(error_map_d, 0.0, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&error_map_block_d, GRIDL * GRIDL * sizeof(source_t)));
    CUDAErrChk(cudaMemset(error_map_block_d, 0.0, GRIDL * GRIDL * sizeof(source_t)));
#endif

    spin_t *XY_mapped_d, *dilution_mask_d;
    energy_t *E_d;

    spin_t *alphas_per_block_d, *block_min_d, *block_max_d, *avg_per_block_d, *local_alphas_d;
    energy_t *T_diluted_per_block_d, *localTemps_d, *intermediate_temps_d;
    unsigned int *acc_local_d;
    CUDAErrChk(cudaMalloc((void **)&T_diluted_per_block_d, GRIDL * GRIDL * sizeof(energy_t)));
    CUDAErrChk(cudaMalloc((void **)&alphas_per_block_d, GRIDL * GRIDL * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&local_alphas_d, N * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&block_min_d, GRIDL * GRIDL * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&block_max_d, GRIDL * GRIDL * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&avg_per_block_d, GRIDL * GRIDL * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void **)&localTemps_d, N * sizeof(energy_t)));
    CUDAErrChk(cudaMalloc((void **)&intermediate_temps_d, N * sizeof(energy_t)));
    CUDAErrChk(cudaMalloc((void **)&acc_local_d, N * sizeof(unsigned int)));
    CUDAErrChk(cudaMemset(acc_local_d, 0, N * sizeof(unsigned int)));
    std::vector<spin_t> local_alphas_h(N);
    


    energy_t *AccD;
    unsigned int* tryD;

    CUDAErrChk(cudaMalloc((void**)&source_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&reconstructed_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&XY_mapped_d, N * sizeof(spin_t)));
    CUDAErrChk(cudaMalloc((void**)&dilution_mask_d, N * sizeof(spin_t)));

    CUDAErrChk(cudaMalloc((void**)&mean_recons_d, N * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&stdDev_recons_d, N * sizeof(source_t)));

    CUDAErrChk(cudaMalloc((void**)&AAE_d, (int)ceil(N / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&ARE_d, (int)ceil(N / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&AARE_d, (int)ceil(N / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&RASE_d, (int)ceil(N / 256.0) * sizeof(source_t)));

    CUDAErrChk(cudaMalloc((void **)&E_d, GRIDL * GRIDL * sizeof(energy_t)));

    CUDAErrChk(cudaMalloc((void**)&AccD, GRIDL * GRIDL * sizeof(energy_t)));
    CUDAErrChk(cudaMalloc((void**)&tryD, GRIDL * GRIDL * sizeof(unsigned int)));

    // for calculating maximum and minimum of data
    source_t *min_d, *max_d;
    CUDAErrChk(cudaMalloc((void**)&min_d, (int)ceil(N / 2.0 / 256.0) * sizeof(source_t)));
    CUDAErrChk(cudaMalloc((void**)&max_d, (int)ceil(N / 2.0 / 256.0) * sizeof(source_t)));

    std::vector<source_t> min_max;

    // copy source data to GPU memory
    CUDAErrChk(cudaMemcpy(source_d, complete_source.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));


#ifdef SOURCE_MAPPING
    // ----- MAPPING PROCESS -----
    std::cout << "------ SOURCE MAPPING PROCESS ------\n";

    min_max_k << < gridLinearLatticeHalf, blockLinearLattice >> > (source_d, min_d, max_d, false, dilution_mask_d);
    CUDAErrChk(cudaPeekAtLastError());

    min_max = findMinMax(min_d, max_d, (int)ceil(N / 2.0 / 256.0));

    std::cout.precision(17);
    std::cout << "from GPU:  min(source) = " << min_max.at(0)
        << " ; max(source) = " << min_max.at(1) << "\n";
    std::cout.precision(6);

    // mapping to XY model based on max and min
    XY_mapping_k << < gridLinearLattice, blockLinearLattice >> > (source_d, XY_mapped_d, min_max.at(0), min_max.at(1), false, dilution_mask_d);
    CUDAErrChk(cudaPeekAtLastError());

    // calculate energy
    energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
    CUDAErrChk(cudaPeekAtLastError());
    energy_t E_source = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;

    // assign temperature
    energy_t T_source = find_temperature(E_source, T_ref, E_ref);
    std::cout << "Source energy per bond: " << E_source << "\n";
    std::cout << "Source temperature: " << T_source << "\n";
#endif

    // print energies
#ifdef ENERGIES_PRINT
    // energies file name + create
    char fileGpuEn[100];
    char fileGpuEnEQ[100];

#ifdef DOUBLE_PRECISION
    sprintf(fileGpuEn, "./data/gpuEn_DP_removed%0.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
    sprintf(fileGpuEnEQ, "./data/gpuEnEQ_DP_removed%0.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
#else
#ifdef INTRINSIC_FLOAT
    sprintf(fileGpuEn, "./data/Energy_SIM_Smooth_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    sprintf(fileGpuEnEQ, "./data/Energy_EQ_Smooth_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
#else
    sprintf(fileGpuEn, "./data/gpuEn_SP_removed%0.3f_Q%0.2f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
    sprintf(fileGpuEnEQ, "./data/gpuEnEQ_removed%0.3f_SP_Q%0.2f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);
#endif
#endif

    FILE *fp = fopen(fileGpuEn, "wb");
    FILE *fpEQ = fopen(fileGpuEnEQ, "wb");
#endif

    // store output data
#ifdef RECONSTRUCTION_PRINT
    char fileMean[100];
    //char fileStdDev[100];

    sprintf(fileMean, "./data/Recons_Smooth_%d_p%0.2f_M%d_%s.dat",
        SMOOTHING_PARAMETER, RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
        
    FILE *fpMean = fopen(fileMean, "wb");
    //FILE *fpStdDev = fopen(fileStdDev, "wb");
#endif

#ifdef ERROR_PRINT
    char fileError[100];
    char fileErrorBlock[100];
    sprintf(fileError, "./data/Error_Smooth_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    sprintf(fileErrorBlock, "./data/Error_Smooth_Block_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    FILE *fpError = fopen(fileError, "wb");
    FILE *fpErrorBlock = fopen(fileErrorBlock, "wb");
#endif

#ifdef TEMPSnALPHAS_PRINT
    char fileTemps_block[100];
    char fileTemps_local[100];
    char fileAlphas_local_eq[100];
    char fileAlphas_local_sim[100];
    sprintf(fileTemps_block, "./data/Temps_Block_Smooth_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    sprintf(fileTemps_local, "./data/Temps_Local_Smooth_%d_p%0.2f_M%d_%s.dat",
        SMOOTHING_PARAMETER, RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    sprintf(fileAlphas_local_eq, "./data/Alphas_Local_EQ_Smooth_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    sprintf(fileAlphas_local_sim, "./data/Alphas_Local_SIM_Smooth_p%0.2f_M%d_%s.dat",
        RemovedDataRatio, CONFIG_SAMPLES, SOURCE_DATA_NAME);
    FILE *fpTempsBlock = fopen(fileTemps_block, "wb");
    FILE *fpTempsLocal = fopen(fileTemps_local, "wb");
    FILE *fpAlphasLocal_Eq = fopen(fileAlphas_local_eq, "wb");
    FILE *fpAlphasLocal_SIM = fopen(fileAlphas_local_sim, "wb");
#endif

#ifdef CONFIGURATION_PRINT
    // print diluted data into file
    spin_t *mask;
    mask = (spin_t*)malloc(N * sizeof(spin_t));
    CUDAErrChk(cudaMemcpy(mask, dilution_mask_d, N * sizeof(spin_t), cudaMemcpyDeviceToHost));
    char strConf[100];
    sprintf(strConf, "./data/conf_removed%1.3f_Q%0.3f_L%d_ConfSamples%d_SwGlob%d.dat",
        RemovedDataRatio, (double)Qfactor, L, CONFIG_SAMPLES, SWEEPS_GLOBAL);

    FILE *f_conf = fopen(strConf, "wb");
#endif

    // SEEDS
    unsigned long long seed_dilution;
    unsigned long long seed_fill;
    unsigned long long seed_simulation;

    // calculation of configurational means
    source_t MAAE = 0.0, MARE = 0.0, MAARE = 0.0, MRASE = 0.0,
        M_timeEQ = 0.0, M_timeSamples = 0.0;
    int sum_eqSw = 0;

    t_geo_end = std::chrono::high_resolution_clock::now();
    long long duration_initial = std::chrono::duration_cast<std::chrono::microseconds>(t_geo_end - t_geo_begin).count();
    long long duration_mapping_EQ_sampling = 0;
    

    /*
    --------------------------------------------------------------
    --------------- LOOP FOR CONFIGURATION SAMPLES ---------------
    --------------------------------------------------------------
    */
    for (int n = 0; n < CONFIG_SAMPLES; ++n)
    {
        // ----- GPU DILUTION ------
        //std::cout << "------ GPU DILUTION ------\n";
        // creating RN generator for dilution
        float *devRand_dil;
        unsigned int *remSum_d;
        CUDAErrChk(cudaMalloc((void **)&devRand_dil, N * sizeof(float)));
        CUDAErrChk(cudaMalloc((void **)&remSum_d, (int)ceil(N / 256.0) * sizeof(unsigned int)));

        curandGenerator_t RNgen_dil;
        curandStatus_t err; // curand errors
        err = curandCreateGenerator(&RNgen_dil, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        cuRAND_ErrChk(err);

        // setting seed
        seed_dilution = (n == 0) ?
#ifdef RNG_SEED_DILUTION 
            RNG_SEED_DILUTION
#else
            time(NULL)
#endif
            : RAN(seed_dilution);

        err = curandSetPseudoRandomGeneratorSeed(RNgen_dil, seed_dilution);
        cuRAND_ErrChk(err);
        // generate random floats on device - for every spin in the lattice and for every local sweep
        err = curandGenerateUniform(RNgen_dil, devRand_dil, N);
        cuRAND_ErrChk(err);


        create_dilution_mask_k << < gridLinearLattice, blockLinearLattice >> > (dilution_mask_d, devRand_dil, remSum_d);
        CUDAErrChk(cudaPeekAtLastError());
        int removedTotal = sumPartialSums(remSum_d, (int)ceil(N / 256.0));

        //std::cout << "Removed = " << removedTotal << " , Removed data ratio = " << removedTotal / (double)N << "\n";

        // RNG cease activity here
        curandDestroyGenerator(RNgen_dil);
        CUDAErrChk(cudaFree(devRand_dil));
        CUDAErrChk(cudaFree(remSum_d));

        // time measurement - relevant part for geostatistical application
        t_geo_begin = std::chrono::high_resolution_clock::now();

        // calculate number of bonds in diluted system
        unsigned int *bondCount_d;

        CUDAErrChk(cudaMalloc((void **)&bondCount_d, GRIDL * GRIDL * sizeof(unsigned int)));
        bondCount_k << < gridEn, blockEn >> > (dilution_mask_d, bondCount_d);
        CUDAErrChk(cudaPeekAtLastError());
        int Nbonds_dil = sumPartialSums(bondCount_d, GRIDL * GRIDL);

        // mapping diluted system to XY model
        min_max_k << < gridLinearLatticeHalf, blockLinearLattice >> > (source_d, min_d, max_d, true, dilution_mask_d);
        CUDAErrChk(cudaPeekAtLastError());

        min_max = findMinMax(min_d, max_d, (int)ceil(N / 2.0 / 256.0));

        /*
        std::cout.precision(17);
        std::cout << "from GPU:  min(diluted) = " << min_max.at(0)
        << " ; max(diluted) = " << min_max.at(1) << "\n";
        std::cout.precision(6);
        */

        // mapping to XY model based on max and min
        XY_mapping_k << < gridLinearLattice, blockLinearLattice >> > (source_d, XY_mapped_d, min_max.at(0), min_max.at(1), true, dilution_mask_d);
        CUDAErrChk(cudaPeekAtLastError());

        // calculate energy
        energyCalcDiluted_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
        CUDAErrChk(cudaPeekAtLastError());
        energy_t E_diluted = sumPartialSums(E_d, (int)GRIDL * GRIDL) / (energy_t)Nbonds_dil;
        // assign a single temperature
        energy_t T_avg = find_temperature(E_diluted, T_ref, E_ref);
        //std::cout << "Diluted - energy per bond: " << E_diluted << "\n";
        //std::cout << "Diluted - temperature: " << T_diluted << "\n";
        //calculate the energy per bond separately in each block

        //number of per-block results
        const int size = GRIDL * GRIDL;
        int n_blocks = std::min((size + 1024 - 1) / 1024, 1024);
        energyCalcDiluted_per_block << <n_blocks, 1024 >> > (E_d, bondCount_d, size);
       // std::cout << "avg bonds per block = " << (energy_t)Nbonds_dil / size << "\n";
        CUDAErrChk(cudaFree(bondCount_d));

        min_max_avg_block << < gridEn, blockEn >> > (XY_mapped_d, block_min_d, block_max_d, avg_per_block_d);
        // assign a different temperature to each block
        find_temperature_gpu << <n_blocks, 1024 >> > (E_d, T_ref_d, E_ref_d, T_diluted_per_block_d, size, ref_size);
        energy_t T_diluted, T_median;
        T_median = find_median(T_diluted_per_block_d, size);
        /*
        energy_t* d_sorted_temps;
        CUDAErrChk(cudaMalloc((void **)&d_sorted_temps, size * sizeof(energy_t)));
        const int num_threads = std::min(size, 1024);
        const int items_per_thread = std::max(size / 1024, 1);
        //sorting block temperatures for finding median block temperature and minimum block temperature
        BlockSortKernel<num_threads, items_per_thread> << <1, num_threads >> >(T_diluted_per_block_d, d_sorted_temps);
        energy_t value1, value2, minT;
        cudaMemcpy(&value1, &d_sorted_temps[(size - 1) / 2], sizeof(energy_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&value2, &d_sorted_temps[size / 2], sizeof(energy_t), cudaMemcpyDeviceToHost);
        T_median = (value1 + value2) / 2;
        //correcting block temperatures in case there were no bond in some blocks, these blocks get the median temperature instead
        correctTemps << <n_blocks, 1024 >> > (T_diluted_per_block_d, T_median);
        //sorting again after correction
        BlockSortKernel<num_threads, items_per_thread> << <1, num_threads >> >(T_diluted_per_block_d, d_sorted_temps);
        cudaMemcpy(&value1, &d_sorted_temps[(size - 1) / 2], sizeof(energy_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&value2, &d_sorted_temps[size / 2], sizeof(energy_t), cudaMemcpyDeviceToHost);
        T_median = (value1 + value2) / 2;
        cudaMemcpy(&minT, &d_sorted_temps[0], sizeof(energy_t), cudaMemcpyDeviceToHost);
        */
        T_diluted = T_median;

        //CUDAErrChk(cudaFree(d_sorted_temps));
        //smoothing block temperatures, creating local temperatures
        smoothTemps << <gridEn, blockEn >> > (T_diluted_per_block_d, intermediate_temps_d, localTemps_d, true);

        for (unsigned int i = 0; i < SMOOTHING_PARAMETER; i++)
        {
            smoothTemps << <gridEn, blockEn >> > (T_diluted_per_block_d, intermediate_temps_d, localTemps_d, false);
            CUDAErrChk(cudaMemcpy(intermediate_temps_d, localTemps_d, N * sizeof(energy_t), cudaMemcpyDeviceToDevice));
            
        }


        if (n == 0)
        {
            std::cout << "Diluted - energy per bond: " << E_diluted << "\n";
            std::cout << "Diluted - temperature: " << T_avg << "\n";
            //std::cout << "Median temperature = " << T_median << "\n";
            //std::cout << "minimum temperature = " << minT << "\n";
            /*
            energy_t* h_sorted_temps;
            cudaMemcpy(h_sorted_temps, d_sorted_temps, size * sizeof(energy_t), cudaMemcpyDeviceToHost);
            CUDAErrChk(cudaFree(d_sorted_temps));
            std::cout << "median temperature CPU = " << T_diluted << " median temperature GPU = " << (value1 + value2) / 2 << "\n\n";
            //std::cout << " median temperature GPU = " << (value1 + value2) / 2 << "\n";

            for (int i = 0; i < size; i++)
            {
            std::cout << h_sorted_temps[i] << "\t";
            }
            std::cout << "\n\n";
            */

#ifdef TEMPSnALPHAS_PRINT
            std::vector<energy_t> blockTemps_h;
            std::vector<energy_t> localTemps_h;
            blockTemps_h.resize(GRIDL * GRIDL);
            localTemps_h.resize(L * L);
            CUDAErrChk(cudaMemcpy(blockTemps_h.data(), T_diluted_per_block_d, sizeof(energy_t) * GRIDL * GRIDL, cudaMemcpyDeviceToHost));
            CUDAErrChk(cudaMemcpy(localTemps_h.data(), localTemps_d, sizeof(energy_t) * L * L, cudaMemcpyDeviceToHost));

            for (int k = 0; k < N; ++k)
                fwrite(&(localTemps_h.at(k)), sizeof(energy_t), 1, fpTempsLocal);

            
            for (int k = 0; k < GRIDL * GRIDL; k++)
            {
                fwrite(&(blockTemps_h.at(k)), sizeof(energy_t), 1, fpTempsBlock);         
            }
            //std::cout << "END OF ITERATION " << n << "\n\n";
                        
#endif
        }
        
#ifdef CONFIGURATION_PRINT
            // print diluted data into file
            spin_t *mask;
            mask = (spin_t*)malloc(N * sizeof(spin_t));
            CUDAErrChk(cudaMemcpy(mask, dilution_mask_d, N * sizeof(spin_t), cudaMemcpyDeviceToHost));

            for (int i = 0; i < N; ++i)
            {
                source_t temp = complete_source.at(i) * mask[i];
                fwrite(&temp, sizeof(source_t), 1, f_conf);
            }
            //std::cout << "I'm here after writing" << "\n\n";
#endif
        
        // ------ FILLING NAN VALUES WITH RANDOM SPINS ------
        //std::cout << "------ FILLING NAN VALUES WITH RANDOM SPINS ------\n";
        // creating RN generator for dilution
                
#ifdef RANDOM_INIT
        float *devRand_fill;
        CUDAErrChk(cudaMalloc((void **)&devRand_fill, N * sizeof(float)));

        curandGenerator_t RNgen_fill;
        err = curandCreateGenerator(&RNgen_fill, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        cuRAND_ErrChk(err);

        // setting seed
        seed_fill = (n == 0) ?
#ifdef RNG_SEED_FILL 
            RNG_SEED_FILL
#else
            time(NULL)
#endif
            : RAN(seed_fill);

        err = curandSetPseudoRandomGeneratorSeed(RNgen_fill, seed_fill);
        cuRAND_ErrChk(err);
        // generate random floats on device - for every spin site
        err = curandGenerateUniform(RNgen_fill, devRand_fill, N);
        cuRAND_ErrChk(err);
        fill_lattice_nans_random << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, devRand_fill);
        // RNG cease activity here
        curandDestroyGenerator(RNgen_fill);
        CUDAErrChk(cudaFree(devRand_fill));
#else
        //spin_t global_average = sumPartialSums(avg_per_block_d, (int)GRIDL * GRIDL) / (GRIDL * GRIDL);
        //fill_lattice_nans_averaged_global << < gridEn, blockEn >> > (XY_mapped_d, global_average);
        fill_lattice_nans_averaged << < gridEn, blockEn >> > (XY_mapped_d, avg_per_block_d);
#endif 
        CUDAErrChk(cudaPeekAtLastError());          

        // ------ CONDITIONED MC SIMULATION -----
        //std::cout << "------ GPU CONDITIONED MC SIMULATION ------\n";
        // create data arrays for thermodynamic variables
        std::vector<energy_t> EnergiesEq;
        std::vector<energy_t> Energies(SWEEPS_GLOBAL);

        // creating RN generator for equilibration and simulation
        // setting seed
        seed_simulation = (n == 0) ?
#ifdef RNG_SEED_SIMULATION 
            RNG_SEED_SIMULATION
#else
            time(NULL)
#endif
            : RAN(seed_simulation);

        // creating RN generator for equilibration and simulation
        float* devRand;
        energy_t alpha = (energy_t)2 * M_PI;

        //std::cout << "iteracia " << n <<"\n";
        
        CUDAErrChk(cudaMalloc((void **)&devRand, 2 * N * SWEEPS_EMPTY * sizeof(float)));

        curandGenerator_t RNgen;
        err = curandCreateGenerator(&RNgen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        cuRAND_ErrChk(err);
        err = curandSetPseudoRandomGeneratorSeed(RNgen, seed_simulation);
        cuRAND_ErrChk(err);

        // summation of reconstructed data for means and standard deviations
        std::vector<source_t> mean_reconstructed(N, 0.0);
        std::vector<source_t> stdDev_reconstructed(N, 0.0);
        CUDAErrChk(cudaMemcpy(mean_recons_d, mean_reconstructed.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));
        CUDAErrChk(cudaMemcpy(stdDev_recons_d, stdDev_reconstructed.data(), N * sizeof(source_t), cudaMemcpyHostToDevice));

        // event creation
        cudaEvent_t start, stop, startEq, stopEq;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&startEq);
        cudaEventCreate(&stopEq);
        float Etime;
        float EtimeEq;

        // start measurment

        cudaEventRecord(startEq, 0);

        // ------ EQUILIBRATION ------
        //std::cout << "------ EQUILIBRATION ------\n";
        // acceptance rate + adjustment of spin-perturbation interval parameter "alpha"
        //float alpha = (float)(2.0*M_PI);
        //double AccRate;

        std::vector<energy_t> AccH(GRIDL * GRIDL, 0.0);
        std::vector<unsigned int> tryH(GRIDL * GRIDL, 0);
        setInitialLocalAlphas << < gridLinearLattice, blockLinearLattice >> > (local_alphas_d, dilution_mask_d);
        //reset accepted flips
        cudaMemset(acc_local_d, 0, N * sizeof(unsigned int));
        //std::vector<energy_t> alphas_H(GRIDL * GRIDL, (energy_t)(2.0*M_PI));
        CUDAErrChk(cudaMemcpy(AccD, AccH.data(), GRIDL * GRIDL * sizeof(energy_t), cudaMemcpyHostToDevice));
        CUDAErrChk(cudaMemcpy(tryD, tryH.data(), GRIDL * GRIDL * sizeof(unsigned int), cudaMemcpyHostToDevice));
        //CUDAErrChk(cudaMemcpy(alphas_per_block_d, alphas_H.data(), GRIDL * GRIDL * sizeof(energy_t), cudaMemcpyHostToDevice));

#ifdef ENERGIES_PRINT
        // Calculate initial energy and write it into file
        energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
        CUDAErrChk(cudaPeekAtLastError());
        energy_t energy = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;
        //EnergiesEq.push_back(energy);

        fwrite(&(energy), sizeof(energy_t), 1, fpEQ);
#endif


        // slope of simple linear regression
        energy_t Slope = -1;
        int it_EQ = 1;
        energy_t meanX = EQUI_TEST_SAMPLES / (energy_t)2.0;
        energy_t varX = 0.0;
        std::vector<energy_t> Xdiff;
        for (int i = 0; i < EQUI_TEST_SAMPLES; ++i)
        {
            Xdiff.push_back(i - meanX);
            varX += Xdiff.at(i) * Xdiff.at(i);
            //std::cout << "Xdiff[i] =  " << Xdiff[i] << "\n";
        }

        Slope = -1;

        while ((Slope < 0) && (it_EQ <= SWEEPS_EQUI_MAX))
        {
            
#ifdef OVER_RELAXATION_EQ
            // over-relaxation algorithm
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 0);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 1);
            CUDAErrChk(cudaPeekAtLastError());
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, 1 / Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
#endif

            err = curandGenerateUniform(RNgen, devRand, 2 * N * SWEEPS_EMPTY);
            cuRAND_ErrChk(err);
            for (int j = 0; j < SWEEPS_EMPTY; ++j)
            {
                metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 0, localTemps_d, acc_local_d, local_alphas_d);
                CUDAErrChk(cudaPeekAtLastError());
                metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 1, localTemps_d, acc_local_d, local_alphas_d);
                CUDAErrChk(cudaPeekAtLastError());
            }
            //std::cout << "it_EQ = " << it_EQ << "\n";
            setLocalAlphas << < gridEn, blockEn >> > (acc_local_d, local_alphas_d, it_EQ, ACC_RATE_MIN_EQ);
            CUDAErrChk(cudaPeekAtLastError());
            // energy calculation and sample filling
            energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
            CUDAErrChk(cudaPeekAtLastError());
            energy_t energy = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;
            EnergiesEq.push_back(energy);
#ifdef ENERGIES_PRINT
            fwrite(&(EnergiesEq.back()), sizeof(energy_t), 1, fpEQ);
#endif

            // keeps the number of energy samples stable (= EQUI_TEST_SAMPLES)
            if (EnergiesEq.size() > EQUI_TEST_SAMPLES)
                EnergiesEq.erase(EnergiesEq.begin());

            ++it_EQ;	// iterator update ("it_EQ = 1" for 1st hybrid sweep)
                        //std::cout << "iteration_global number " << it_EQ << " has energy " << energy <<"\n";
                        // Slope update
            if ((it_EQ % EQUI_TEST_FREQUENCY) == 0)
            {
                // testing equilibration condition - claculation of linear regression slope from stored energies
                if (EnergiesEq.size() == EQUI_TEST_SAMPLES)
                {
                    energy_t sumEn = 0.0;
                    for (auto n : EnergiesEq) sumEn += n;
                    energy_t meanEn = sumEn / EQUI_TEST_SAMPLES;
                    sumEn = 0.0;
                    for (int k = 0; k < EQUI_TEST_SAMPLES; ++k)
                        sumEn += (EnergiesEq.at(k) - meanEn) * Xdiff.at(k);
                    Slope = sumEn / varX;
                    //std::cout << "Slope_global =  " << Slope << "\n";
                }
            }
        }

        // end measurment
        CUDAErrChk(cudaEventRecord(stopEq, 0));
        CUDAErrChk(cudaEventSynchronize(stopEq));
        CUDAErrChk(cudaEventElapsedTime(&EtimeEq, startEq, stopEq));

#ifdef ENERGIES_PRINT
        for (int i = it_EQ; i <= SWEEPS_EQUI_MAX; i++)
        {
            int k = 0;
            fwrite(&k, sizeof(int), 1, fpEQ);
        }
#endif

#ifdef TEMPSnALPHAS_PRINT
        if (n == 0)
        {
            local_alphas_h.resize(N);
            CUDAErrChk(cudaMemcpy(local_alphas_h.data(), local_alphas_d, sizeof(spin_t) * N, cudaMemcpyDeviceToHost));

            for (int k = 0; k < N; ++k)
                fwrite(&(local_alphas_h.at(k)), sizeof(spin_t), 1, fpAlphasLocal_Eq);
            //std::cout << "lokalna alfa 4 = " << local_alphas_h[4] << "\n";
        }
#endif 
        // start measurment

        cudaEventRecord(start, 0);

        // ------ GENERATING SAMPLES ------
        //single checkerboard version
        for (int i = 0; i < SWEEPS_GLOBAL; ++i)
        {
            
#ifdef OVER_RELAXATION_SIM
            // over-relaxation algorithm
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 0);
            CUDAErrChk(cudaPeekAtLastError());
            over_relaxation_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, 1);
            CUDAErrChk(cudaPeekAtLastError());
            spin_mult << < gridLinearLattice, blockLinearLattice >> > (XY_mapped_d, 1 / Qfactor);
            CUDAErrChk(cudaPeekAtLastError());
#endif
            // generate random floats on device - for every spin in the lattice and for every empty sweep
            err = curandGenerateUniform(RNgen, devRand, 2 * N * SWEEPS_EMPTY);
            cuRAND_ErrChk(err);

            for (int j = 0; j < SWEEPS_EMPTY; ++j)
            {
                metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 0, localTemps_d, acc_local_d, local_alphas_d);
                CUDAErrChk(cudaPeekAtLastError());
                metro_conditioned_sublattice_k << < grid_check, block_check >> > (XY_mapped_d, dilution_mask_d, devRand + j*N, 1, localTemps_d, acc_local_d, local_alphas_d);
                CUDAErrChk(cudaPeekAtLastError());
            }
            setLocalAlphas << < gridEn, blockEn >> > (acc_local_d, local_alphas_d, it_EQ + i, ACC_RATE_MIN_EQ); 
#ifdef ENERGIES_PRINT
            // energy calculation
            energyCalc_k << < gridEn, blockEn >> > (XY_mapped_d, E_d);
            CUDAErrChk(cudaPeekAtLastError());
            Energies.at(i) = sumPartialSums(E_d, (int)GRIDL * GRIDL) / Nbond;
            fwrite(&(Energies.at(i)), sizeof(energy_t), 1, fp);
#endif

            // data reconstruction + summation for mean and standard deviation
            data_reconstruction_k << < gridLinearLattice, blockLinearLattice >> > (reconstructed_d, XY_mapped_d, min_max.at(0), min_max.at(1), mean_recons_d, stdDev_recons_d);
            CUDAErrChk(cudaPeekAtLastError());
        }

        // end measurment
        CUDAErrChk(cudaEventRecord(stop, 0));
        CUDAErrChk(cudaEventSynchronize(stop));
        CUDAErrChk(cudaEventElapsedTime(&Etime, start, stop));

        // GPU time
        M_timeEQ += EtimeEq / 1000;
        M_timeSamples += Etime / 1000;

        // prediction averages and standard deviations
        mean_stdDev_reconstructed_k << < gridLinearLattice, blockLinearLattice >> > (mean_recons_d, stdDev_recons_d, SWEEPS_GLOBAL);
        CUDAErrChk(cudaPeekAtLastError());

        t_geo_end = std::chrono::high_resolution_clock::now();
        duration_mapping_EQ_sampling += std::chrono::duration_cast<std::chrono::microseconds>(t_geo_end - t_geo_begin).count();

#ifdef RECONSTRUCTION_PRINT
        CUDAErrChk(cudaMemcpy(mean_reconstructed.data(), mean_recons_d, N * sizeof(source_t), cudaMemcpyDeviceToHost));
        if (n == 0)
        {
            for (int k = 0; k < N; ++k)
            {
                //std::cout << "zapisujem " << mean_reconstructed.at(k) << " a " << stdDev_reconstructed.at(k) << "\n";
                fwrite(&(mean_reconstructed.at(k)), sizeof(source_t), 1, fpMean);
                //fwrite(&(stdDev_reconstructed.at(k)), sizeof(source_t), 1, fpStdDev);
            }
        }
#endif
       
        // does not work with BlockL 8
#ifdef ERROR_PRINT
        // prediction errors
        sum_prediction_errors_k << < gridEn, blockEn >> > (source_d, mean_recons_d, dilution_mask_d, AAE_d, ARE_d, AARE_d, RASE_d, error_map_d, error_map_block_d);
        CUDAErrChk(cudaPeekAtLastError());
        MAAE += sumPartialSums(AAE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal;
        MARE += sumPartialSums(ARE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal;
        MAARE += sumPartialSums(AARE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal;
        MRASE += sqrt(sumPartialSums(RASE_d, (int)GRIDL * GRIDL) / (source_t)removedTotal);
#else
        // works with BlockL 8
        // prediction errors
        sum_prediction_errors_k << < gridLinearLattice, blockLinearLattice >> > (source_d, mean_recons_d, dilution_mask_d, AAE_d, ARE_d, AARE_d, RASE_d);
        CUDAErrChk(cudaPeekAtLastError());
        MAAE += sumPartialSums(AAE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal;
        MARE += sumPartialSums(ARE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal;
        MAARE += sumPartialSums(AARE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal;
        MRASE += sqrt(sumPartialSums(RASE_d, (int)ceil(N / 256.0)) / (source_t)removedTotal);
#endif
        

        // Number of equilibration sweeps
        sum_eqSw += it_EQ;

        // cudaFree after equilibration
        curandDestroyGenerator(RNgen);
        CUDAErrChk(cudaFree(devRand));
        

        if (n == 0) std::cout << "Seeds[configurations, filling, simulation] = " << "["
            << seed_dilution << ", " << seed_fill << ", " << seed_simulation << "]\n";

    }

    std::cout.precision(8);
    std::cout << "Mean elapsed time (equilibration for average " << sum_eqSw / (source_t)CONFIG_SAMPLES << " sweeps) = " << M_timeEQ / CONFIG_SAMPLES << " s\n";
    std::cout << "Mean elapsed time (collection of " << CONFIG_SAMPLES << " samples) = " << M_timeSamples / CONFIG_SAMPLES << " s\n";

    // prediction errors
    std::cout << "MAAE = " << MAAE / CONFIG_SAMPLES << "\n";
    std::cout << "MARE = " << MARE * 100 / CONFIG_SAMPLES << " %\n";
    std::cout << "MAARE = " << MAARE * 100 / CONFIG_SAMPLES << " %\n";
    std::cout << "MRASE = " << MRASE / CONFIG_SAMPLES << "\n";


#ifdef ERROR_PRINT
    std::vector<source_t> error_map_h(N);
    std::vector<source_t> error_map_block_h(GRIDL * GRIDL);
    CUDAErrChk(cudaMemcpy(error_map_h.data(), error_map_d, N * sizeof(source_t), cudaMemcpyDeviceToHost));
    CUDAErrChk(cudaMemcpy(error_map_block_h.data(), error_map_block_d, GRIDL * GRIDL * sizeof(source_t), cudaMemcpyDeviceToHost));

    for (int k = 0; k < N; ++k)
    {
        error_map_h[k] = error_map_h[k] / (source_t)CONFIG_SAMPLES;
        fwrite(&(error_map_h.at(k)), sizeof(source_t), 1, fpError);
    }
    
    for (int k = 0; k < GRIDL * GRIDL; k++)
    {
        error_map_block_h[k] = error_map_block_h[k] / (source_t)CONFIG_SAMPLES;
        fwrite(&(error_map_block_h.at(k)), sizeof(source_t), 1, fpErrorBlock);
    }
#endif

    // closing time series storage
#ifdef ENERGIES_PRINT  
    fclose(fp);
    fclose(fpEQ);
#endif
#ifdef RECONSTRUCTION_PRINT
    fclose(fpMean);
    //fclose(fpStdDev);
#endif
#ifdef CONFIGURATION_PRINT
    fclose(f_conf);
#endif
#ifdef ERROR_PRINT
    CUDAErrChk(cudaFree(error_map_d));
    fclose(fpError);
    CUDAErrChk(cudaFree(error_map_block_d));
    fclose(fpErrorBlock);
#endif
#ifdef TEMPSnALPHAS_PRINT
    CUDAErrChk(cudaMemcpy(local_alphas_h.data(), local_alphas_d, sizeof(spin_t) * N, cudaMemcpyDeviceToHost));

    for (int k = 0; k < N; ++k)
        fwrite(&(local_alphas_h.at(k)), sizeof(spin_t), 1, fpAlphasLocal_SIM);
    
    fclose(fpTempsBlock);
    fclose(fpTempsLocal);
    fclose(fpAlphasLocal_Eq);
    fclose(fpAlphasLocal_SIM);
#endif 

    // free CUDA variable
    CUDAErrChk(cudaFree(source_d));
    CUDAErrChk(cudaFree(XY_mapped_d));
    CUDAErrChk(cudaFree(dilution_mask_d));
    CUDAErrChk(cudaFree(reconstructed_d));
    CUDAErrChk(cudaFree(min_d));
    CUDAErrChk(cudaFree(max_d));
    CUDAErrChk(cudaFree(E_d));


    CUDAErrChk(cudaFree(T_diluted_per_block_d));
    CUDAErrChk(cudaFree(alphas_per_block_d));
    CUDAErrChk(cudaFree(T_ref_d));
    CUDAErrChk(cudaFree(E_ref_d));
    CUDAErrChk(cudaFree(tryD));
    CUDAErrChk(cudaFree(block_min_d));
    CUDAErrChk(cudaFree(block_max_d));
    CUDAErrChk(cudaFree(avg_per_block_d));
    CUDAErrChk(cudaFree(localTemps_d));
    CUDAErrChk(cudaFree(intermediate_temps_d));
    CUDAErrChk(cudaFree(local_alphas_d));
    CUDAErrChk(cudaFree(acc_local_d));


    CUDAErrChk(cudaFree(AccD));
    CUDAErrChk(cudaFree(mean_recons_d));
    CUDAErrChk(cudaFree(stdDev_recons_d));
    CUDAErrChk(cudaFree(AAE_d));
    CUDAErrChk(cudaFree(ARE_d));
    CUDAErrChk(cudaFree(AARE_d));
    CUDAErrChk(cudaFree(RASE_d));

    // time measurement - entire process
    std::chrono::high_resolution_clock::time_point t_sim_end = std::chrono::high_resolution_clock::now();
    auto tot_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_sim_end - t_sim_begin).count();
    std::cout << "Total duration = " << (double)tot_duration / 1e6 << " s\n";
    std::cout << "Total duration per configuration sample = " << (double)tot_duration / 1e6 / CONFIG_SAMPLES << " s\n";
    // time measurement - relevant part for geostatistical application
    //(loading reference E = E(T), loading source, mapping to XY model, equilibration and reconstruction sample collection)
    std::cout << "------DURATION OF GEOSTATISTICAL APPLICATION------\n"
        //<< "Inicialization processes (loading reference E=E(T), loading source data, GPU memory allocation and copying):\n"
        << "t_initialization = " << (double)duration_initial / 1e6 << " s\n"
        //<< "Mapping to XY model, equilibration and reconstruction sample collection (per configuration sample):\n"
        << "t_reconstruction = " << (double)duration_mapping_EQ_sampling / 1e6 / CONFIG_SAMPLES << " s\n"
        //<< "Mapping to XY model, equilibration and reconstruction sample collection:\n"
        << "t_TOTAL = " << ((double)duration_initial / 1e6 + (double)duration_mapping_EQ_sampling / 1e6 / CONFIG_SAMPLES) << " s\n";

    return 0;
}

//single checkerboard versions
__global__ void metro_conditioned_sublattice_k(spin_t *s, spin_t *dilution_mask_d, float *devRand, unsigned int offset, energy_t* localTemps, unsigned int *AccD, spin_t* local_alphas_d)
{
    // int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = 2 * threadIdx.y + (threadIdx.x + offset) % 2 + BLOCKL*blockIdx.y;

    unsigned int n = threadIdx.x + threadIdx.y*BLOCKL;
    unsigned int idx = n + THREADS * (blockIdx.x + gridDim.x*blockIdx.y);
    
    if (isnan(dilution_mask_d[x + L*y]))
    {
        energy_t BETA = 1.0 / localTemps[x + L*y];
        spin_t S_old = s[x + L*y];
        spin_t S_new = S_old + local_alphas_d[x + L*y] * (devRand[idx + offset*N / 2 + N*SWEEPS_EMPTY] - 0.5f);
        //spin_t S_new = S_old + (energy_t) (2.0f * M_PI) * (devRand[idx + offset*N / 2 + N*SWEEPS_EMPTY] - 0.5f);
        S_new = (S_new < 0.0f) ? 0.0f : S_new;
        S_new = (S_new > 2.0f * M_PI) ? 2.0f * M_PI : S_new;

        energy_t E1 = 0.0, E2 = 0.0;

        // NOTE: open boundary conditions -> energy contribution on boundary always results in -cos(S(x,y) - S(x,y)) = -1 
#ifdef DOUBLE_PRECISION
        E1 -= (x == 0) ? 1 : cos(Qfactor * (S_old - s[x - 1 + L*y]));		// from s(x-1,y)
        E2 -= (x == 0) ? 1 : cos(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : cos(Qfactor * (S_old - s[x + 1 + L*y]));	// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : cos(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : cos(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : cos(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : cos(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : cos(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < exp(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#else
#ifdef INTRINSIC_FLOAT
        E1 -= (x == 0) ? 1 : __cosf(Qfactor * (S_old - s[x - 1 + L*y]));		// from s(x-1,y)
        E2 -= (x == 0) ? 1 : __cosf(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : __cosf(Qfactor * (S_old - s[x + 1 + L*y]));	// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : __cosf(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : __cosf(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : __cosf(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : __cosf(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : __cosf(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < __expf(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            AccD[x + L*y] = AccD[x + L*y] + 1;
            //if (x + L*y == 35)
                //printf("thread %d local alpha = % 1.5f,  acc_local = %d\n", x + L*y, local_alphas_d[x + L*y], AccD[x + L*y]);
        }
#else
        E1 -= (x == 0) ? 1 : cosf(Qfactor * (S_old - s[x - 1 + L*y]));			// from s(x-1,y)
        E2 -= (x == 0) ? 1 : cosf(Qfactor * (S_new - s[x - 1 + L*y]));
        E1 -= (x == L - 1) ? 1 : cosf(Qfactor * (S_old - s[x + 1 + L*y]));		// from s(x+1,y)
        E2 -= (x == L - 1) ? 1 : cosf(Qfactor * (S_new - s[x + 1 + L*y]));
        E1 -= (y == 0) ? 1 : cosf(Qfactor * (S_old - s[x + L*(y - 1)]));		// from s(x,y-1)
        E2 -= (y == 0) ? 1 : cosf(Qfactor * (S_new - s[x + L*(y - 1)]));
        E1 -= (y == L - 1) ? 1 : cosf(Qfactor * (S_old - s[x + L*(y + 1)]));	// from s(x,y+1)
        E2 -= (y == L - 1) ? 1 : cosf(Qfactor * (S_new - s[x + L*(y + 1)]));

        if (devRand[idx + offset*N / 2] < expf(-BETA * (E2 - E1)))
        {
            s[x + L*y] = S_new;
            ++Acc;
        }
#endif
#endif

    }    
}

__global__ void spin_mult(spin_t *s, spin_t mult_factor)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;

    s[idx] = s[idx] * mult_factor;
}

__global__ void over_relaxation_k(spin_t *s, spin_t *dilution_mask_d, int offset)
{
    // int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = 2 * threadIdx.y + (threadIdx.x + offset) % 2 + BLOCKL*blockIdx.y;

    energy_t sumSin = 0.0, sumCos = 0.0;
    spin_t s_new;

    // checkerboard update
    // not updating spins on the edge of the system
    if (isnan(dilution_mask_d[x + L*y]) && (x > 0) && (x < L - 1) && (y > 0) && (y < L - 1))
    {
        //summation of sin and cos from neighbouring spins
#ifdef DOUBLE_PRECISION
        sumSin += sin(s[x - 1 + L*y]);
        sumCos += cos(s[x - 1 + L*y]);
        sumSin += sin(s[x + 1 + L*y]);
        sumCos += cos(s[x + 1 + L*y]);
        sumSin += sin(s[x + L*(y - 1)]);
        sumCos += cos(s[x + L*(y - 1)]);
        sumSin += sin(s[x + L*(y + 1)]);
        sumCos += cos(s[x + L*(y + 1)]);
#else
#ifdef INTRINSIC_FLOAT
        sumSin += __sinf(s[x - 1 + L*y]);
        sumCos += __cosf(s[x - 1 + L*y]);
        sumSin += __sinf(s[x + 1 + L*y]);
        sumCos += __cosf(s[x + 1 + L*y]);
        sumSin += __sinf(s[x + L*(y - 1)]);
        sumCos += __cosf(s[x + L*(y - 1)]);
        sumSin += __sinf(s[x + L*(y + 1)]);
        sumCos += __cosf(s[x + L*(y + 1)]);
#else
        sumSin += sinf(s[x - 1 + L*y]);
        sumCos += cosf(s[x - 1 + L*y]);
        sumSin += sinf(s[x + 1 + L*y]);
        sumCos += cosf(s[x + 1 + L*y]);
        sumSin += sinf(s[x + L*(y - 1)]);
        sumCos += cosf(s[x + L*(y - 1)]);
        sumSin += sinf(s[x + L*(y + 1)]);
        sumCos += cosf(s[x + L*(y + 1)]);
#endif
#endif
        s_new = (spin_t)(fmod(2.0 * atan2(sumSin, sumCos) - s[x + L*y], 2.0 * M_PI));
        if ((s_new >= 0.0) && (s_new <= Qfactor * 2 * M_PI))
            s[x + L*y] = s_new;
    }
}

__global__ void energyCalc_k(spin_t *s, energy_t *Ed) {

    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;

    energy_t partE = 0;

    // (x,y < L - 1) conditions prevent from accounting bonds outside system boundaries 
#ifdef DOUBLE_PRECISION
    // if (x < L - 1) partE -= cos((energy_t)(Qfactor * (s[x + L*y] - s[x + 1 + L*y])));
    // if (y < L - 1) partE -= cos((energy_t)(Qfactor * (s[x + L*y] - s[x + L*(y + 1)])));
    if (x < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
    if (y < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#else

#ifdef INTRINSIC_FLOAT
    if (x < L - 1) partE -= __cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
    if (y < L - 1) partE -= __cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#else
    if (x < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
    if (y < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#endif

#endif

    __shared__ energy_t EnSum[BLOCKL*BLOCKL];
    EnSum[t] = partE;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) EnSum[t] += EnSum[t + stride];
    }

    if (t == 0) Ed[blockIdx.x + gridDim.x*blockIdx.y] = EnSum[0];

}

__global__ void energyCalcDiluted_k(spin_t *s, energy_t *Ed)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;

    energy_t partE = 0;
    energy_t tryLocalE;


    // (x,y < L - 1) conditions prevent from accounting bonds outside system boundaries 
#ifdef DOUBLE_PRECISION	
    if (x < L - 1)
    {
        tryLocalE = cos(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
    if (y < L - 1)
    {
        tryLocalE = cos(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
#else

#ifdef INTRINSIC_FLOAT
    if (x < L - 1)
    {
        tryLocalE = __cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
    if (y < L - 1)
    {
        tryLocalE = __cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
#else
    if (x < L - 1)
    {
        tryLocalE = cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
    if (y < L - 1)
    {
        tryLocalE = cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
        partE -= isnan(tryLocalE) ? 0 : tryLocalE;
    }
#endif

#endif

    __shared__ energy_t EnSum[BLOCKL*BLOCKL];
    EnSum[t] = partE;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) EnSum[t] += EnSum[t + stride];
    }

    if (t == 0) Ed[blockIdx.x + gridDim.x*blockIdx.y] = EnSum[0];
    //uncomment for verifying the correct simulation temperatures per block
    //verification[x + L*y] = blockIdx.x + gridDim.x*blockIdx.y;
}

__global__ void energyCalcDiluted_per_block(energy_t *Ed, unsigned int *bondCount_d, unsigned int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        Ed[i] = Ed[i] / bondCount_d[i];
        //if(Ed[i] > - 0.4592)
          //  printf("energy in block %d is %1.2f and bond count is %d\n", i, Ed[i], bondCount_d[i]);
    }
}

__global__ void min_max_avg_block(spin_t *d_s, spin_t *d_min, spin_t *d_max, spin_t *d_avg)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;


    //stores values needed to compute min, max, sum and number of non-NaN values
    __shared__ spin_t min_max_avg_s[4 * BLOCKL*BLOCKL];
    spin_t spin = d_s[x + L*y];
    //if(t == 0)
    //printf("block %d has number = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, d_s[x + L*y]);

    min_max_avg_s[t] = spin;
    min_max_avg_s[t + BLOCKL*BLOCKL] = spin;
    min_max_avg_s[t + 2 * BLOCKL*BLOCKL] = isnan(spin) ? 0 : spin;
    min_max_avg_s[t + 3 * BLOCKL*BLOCKL] = isnan(spin) ? 0 : 1;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride)
        {
            min_max_avg_s[t] = fmin(min_max_avg_s[t], min_max_avg_s[t + stride]);				// minimum search
            min_max_avg_s[t + BLOCKL*BLOCKL] = fmax(min_max_avg_s[t + BLOCKL*BLOCKL], min_max_avg_s[t + BLOCKL*BLOCKL + stride]);	// maximum search
            min_max_avg_s[t + 2 * BLOCKL*BLOCKL] += min_max_avg_s[t + 2 * BLOCKL*BLOCKL + stride];
            min_max_avg_s[t + 3 * BLOCKL*BLOCKL] += min_max_avg_s[t + 3 * BLOCKL*BLOCKL + stride];
        }
    }

    if (t == 0)
    {
        d_min[blockIdx.x + gridDim.x*blockIdx.y] = min_max_avg_s[0];
        d_max[blockIdx.x + gridDim.x*blockIdx.y] = min_max_avg_s[BLOCKL*BLOCKL];
        d_avg[blockIdx.x + gridDim.x*blockIdx.y] = min_max_avg_s[2 * BLOCKL*BLOCKL] / min_max_avg_s[3 * BLOCKL*BLOCKL];
        //printf("block %d has number = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, d_pointers_to_blocks[blockIdx.x + gridDim.x*blockIdx.y][t]);

        //uncomment for verification
        //if(min_max_avg_s[BLOCKL*BLOCKL] > 6.0)
        //printf("block %d has min = %1.7f and max = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, min_max_avg_s[0], min_max_avg_s[BLOCKL*BLOCKL]);
        //printf("block %d has avg = %1.7f\n", blockIdx.x + gridDim.x*blockIdx.y, avg[blockIdx.x + gridDim.x*blockIdx.y]);

    }
    //uncomment for verification
    /*
    __syncthreads();
    if (blockIdx.x + gridDim.x*blockIdx.y == 179)
    {
    for (int i = 0; i < BLOCKL*BLOCKL; i++)
    {
    __syncthreads();
    if(t == i)
    printf("thread %d has a low = %1.7f and high = %1.7f\n", t, min_max_avg_s[t], min_max_avg_s[t_off]);
    }

    }
    */
}

__global__ void resetAccD_k(energy_t *AccD) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < BLOCKS) AccD[idx] = 0;

}

__global__ void setInitialLocalAlphas(spin_t *local_alphas_d, spin_t *dilution_mask_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;

    if (isnan(dilution_mask_d[idx]))
        local_alphas_d[idx] = (spin_t) 2*M_PI;
    else
        local_alphas_d[idx] = (spin_t) 0.0;
}

__global__ void setBlockAlphas(energy_t *AccD, spin_t *alphas_per_block_d, int iterations, energy_t acc_rate_min, unsigned int* tryD, spin_t* block_min, spin_t* block_max)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BLOCKS * 2)
    {
        energy_t acc_rate = AccD[idx] / (energy_t)tryD[idx];

        if (alphas_per_block_d[idx] > 0)
        {
            if (acc_rate < acc_rate_min)
            {
                alphas_per_block_d[idx] = (block_max[idx] - block_min[idx]) / (1 + iterations / (spin_t)SLOPE_RESTR_FACTOR);
            }
        }

    }
}

__global__ void setLocalAlphas(unsigned int *AccD, spin_t *local_alphas_d, int iterations, energy_t acc_rate_min)
{
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;
    unsigned int idx = x + L*y;
    
    energy_t acc_rate = AccD[idx] / (energy_t)iterations;

        if (local_alphas_d[idx] > 0)
        {
            if (acc_rate < acc_rate_min)
            {
                local_alphas_d[idx] = 2 * M_PI / (1 + iterations / (spin_t)SLOPE_RESTR_FACTOR);
            }
         }    
}

__global__ void min_max_k(source_t *source_d, source_t *min_d, source_t *max_d, bool isDiluted, spin_t *diluted_mask_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + (blockDim.x * 2) * b;

    unsigned int t_off = t + 256;	// shared memory access with offset - it was calculated too many times

    /* By declaring the shared memory buffer as "volatile", the compiler is forced to enforce
    the shared memory write after each stage of the reduction,
    and the implicit data synchronisation between threads within the warp is restored */
    __shared__ volatile source_t min_max_s[512];
    if (isDiluted)
    {
        min_max_s[t] = source_d[idx] * diluted_mask_d[idx];
        min_max_s[t_off] = source_d[idx + 256] * diluted_mask_d[idx + 256];
    }
    else
    {
        min_max_s[t] = source_d[idx];
        min_max_s[t_off] = source_d[idx + 256];
    }

    __syncthreads();

    // divide min_max_avg_s araray to "min" part (indices 0 ... 255) and "max" part (256 ... 511)
    // macros min(a,b) (and max(a,b)) from math.h are equivalent to conditional ((a < b) ? (a) : (b)) -> will be added in preprocessing
    source_t temp = fmax(min_max_s[t], min_max_s[t_off]);
    min_max_s[t] = fmin(min_max_s[t], min_max_s[t_off]);
    min_max_s[t_off] = temp;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128)
    {
        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 128]);				// minimum search
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 128]);	// maximum search
    }

    __syncthreads();
    if (t < 64)
    {
        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 64]);				// minimum search
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 64]);	// maximum search
    }

    /* when we have one warp left ->
    no need for "if(t<stride)" and "__syncthreads"
    (no extra work is saved and because instructions are SIMD synchronous within a warp)	*/
    __syncthreads();
    if (t < 32)
    {
        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 32]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 32]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 16]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 16]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 8]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 8]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 4]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 4]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 2]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 2]);

        min_max_s[t] = fmin(min_max_s[t], min_max_s[t + 1]);
        min_max_s[t_off] = fmax(min_max_s[t_off], min_max_s[t_off + 1]);
    }

    // per block results are stored to global memory
    if (t == 0)
    {
        min_d[b] = min_max_s[0];
        max_d[b] = min_max_s[256];
        //if(min_max_avg_s[BLOCKL*BLOCKL] > 6.0)
        //printf("block %d has min = %1.7f and max = %1.7f\n", b, min_d[b], max_d[b]);
    }
}

__global__ void XY_mapping_k(source_t *source_d, spin_t *XY_mapped_d, source_t minSource, source_t maxSource, bool isDiluted, spin_t *diluted_mask_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;

    XY_mapped_d[idx] = (isDiluted) ? (spin_t)(2 * M_PI * (source_d[idx] * diluted_mask_d[idx] - minSource) / (maxSource - minSource)) :
        (spin_t)(2 * M_PI * (source_d[idx] - minSource) / (maxSource - minSource));
    //if (XY_mapped_d[idx]  > 6.29)
    //printf("block %d has value = %1.7f, maxSource = %1.7f, source_d[%d] = %1.7f\n", b, XY_mapped_d[idx], maxSource, idx, source_d[idx]);

}

__global__ void create_dilution_mask_k(spin_t *dilution_mask_d, float* devRandDil, unsigned int* remSum_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;
    unsigned int rem;
    if (devRandDil[idx] < RemovedDataRatio)
    {
#ifdef DOUBLE_PRECISION
        dilution_mask_d[idx] = nan("");
#else
        dilution_mask_d[idx] = nanf("");
#endif
        rem = 1;
    }
    else
    {
        dilution_mask_d[idx] = 1;
        rem = 0;
    }
    volatile __shared__ unsigned int removed_Sum[256];
    removed_Sum[t] = rem;
    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) removed_Sum[t] += removed_Sum[t + 128];

    __syncthreads();
    if (t < 64) removed_Sum[t] += removed_Sum[t + 64];

    // reduction for last warp
    __syncthreads();
    if (t < 32)
    {
        removed_Sum[t] += removed_Sum[t + 32];
        removed_Sum[t] += removed_Sum[t + 16];
        removed_Sum[t] += removed_Sum[t + 8];
        removed_Sum[t] += removed_Sum[t + 4];
        removed_Sum[t] += removed_Sum[t + 2];
        removed_Sum[t] += removed_Sum[t + 1];
    }

    if (t == 0) remSum_d[b] = removed_Sum[0];
}

__global__ void fill_lattice_nans_averaged(spin_t *XY_mapped_d, spin_t *avg)
{
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;
    unsigned int idx = x + L*y;
    if (isnan(XY_mapped_d[idx])) XY_mapped_d[idx] = avg[blockIdx.x + gridDim.x*blockIdx.y];
}

__global__ void fill_lattice_nans_averaged_global(spin_t *XY_mapped_d, spin_t avg)
{
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;
    unsigned int idx = x + L*y;
    if (isnan(XY_mapped_d[idx])) XY_mapped_d[idx] = avg;
}

__global__ void fill_lattice_nans_random(spin_t *XY_mapped_d, float*devRand_fill)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (isnan(XY_mapped_d[idx])) XY_mapped_d[idx] = 2 * M_PI * devRand_fill[idx];
}

__global__ void data_reconstruction_k(source_t *reconstructed_d, spin_t *XY_mapped_d, source_t minSource, source_t maxSource, source_t *sum_d, source_t *sumSqr_d)
{
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    unsigned int idx = t + blockDim.x * b;

    reconstructed_d[idx] = ((source_t)XY_mapped_d[idx])*(maxSource - minSource) / (2 * M_PI) + minSource;
    sum_d[idx] += reconstructed_d[idx];
    sumSqr_d[idx] += reconstructed_d[idx] * reconstructed_d[idx];
}

__global__ void bondCount_k(spin_t *mask_d, unsigned int *bondCount_d)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;

    unsigned int bondCount = 0;
    bool isNotCentralNAN = !isnan(mask_d[x + L*y]);

    if (x < L - 1)
        bondCount += isNotCentralNAN && (!isnan(mask_d[x + 1 + L*y]));
    if (y < L - 1)
        bondCount += isNotCentralNAN && (!isnan(mask_d[x + L*(y + 1)]));

    __shared__ unsigned int bondSum[BLOCKL*BLOCKL];
    bondSum[t] = bondCount;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) bondSum[t] += bondSum[t + stride];
    }

    if (t == 0) bondCount_d[blockIdx.x + gridDim.x*blockIdx.y] = bondSum[0];
    //if (t == 0 && blockIdx.x + gridDim.x*blockIdx.y == 32) bondCount_d[blockIdx.x + gridDim.x*blockIdx.y] = 0;
    //if (t == 0 && bondSum[0] < 20) printf("bond count in block %d is %d\n", blockIdx.x + gridDim.x*blockIdx.y, bondSum[0]);
}

__global__ void mean_stdDev_reconstructed_k(source_t *mean_d, source_t *stdDev_d, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    mean_d[idx] /= size;
    stdDev_d[idx] = sqrt(stdDev_d[idx] / size + mean_d[idx] * mean_d[idx]);
}

__global__ void sum_prediction_errors_k(source_t *source_d, source_t * mean_d, spin_t *dilution_mask_d,
    source_t *AAE_d, source_t *ARE_d, source_t *AARE_d, source_t *RASE_d, source_t* error_map_d, source_t* error_map_block_d)
{
    unsigned int t = threadIdx.x + BLOCKL*threadIdx.y;
    unsigned int x = threadIdx.x + BLOCKL*blockIdx.x;
    unsigned int y = threadIdx.y + BLOCKL*blockIdx.y;
    unsigned int idx = x + L*y;
    
    source_t source = source_d[idx];
    source_t est_error = (source - mean_d[idx]);
    bool isnan_site = isnan(dilution_mask_d[idx]);

    //get error for each particular spin for error map
    error_map_d[idx] += isnan_site * fabs(est_error);

    volatile __shared__ source_t sum_err[BLOCKL*BLOCKL];
    volatile __shared__ unsigned int validSpins[BLOCKL*BLOCKL];
    // AVERAGE ABSOLUTE ERROR
    sum_err[t] = isnan_site * fabs(est_error);
    validSpins[t] = (unsigned int) isnan_site;
    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride)
        {
            sum_err[t] += sum_err[t + stride];
            validSpins[t] += validSpins[t + stride];
        }
    }

    if (t == 0)
    {
        AAE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];
        error_map_block_d[blockIdx.x + gridDim.x*blockIdx.y] += sum_err[0] / (source_t)validSpins[0];
    }
    // AVERAGE RELAITVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * est_error / source;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) sum_err[t] += sum_err[t + stride];
    }
    if (t == 0) ARE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];
    // AVERAGE ABSOLUTE RELATIVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * fabs(est_error) / source;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) sum_err[t] += sum_err[t + stride];
    }
    if (t == 0) AARE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];

    // summation for ROOT AVERAGE SQUARED ROOT
    __syncthreads();
    sum_err[t] = isnan_site * est_error * est_error;

    for (unsigned int stride = (BLOCKL*BLOCKL) >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (t < stride) sum_err[t] += sum_err[t + stride];
    }
    if (t == 0) RASE_d[blockIdx.x + gridDim.x*blockIdx.y] = sum_err[0];
}

__global__ void sum_prediction_errors_k(source_t *source_d, source_t * mean_d, spin_t *dilution_mask_d,
    source_t *AAE_d, source_t *ARE_d, source_t *AARE_d, source_t *RASE_d)
{
    unsigned int t = threadIdx.x;
    unsigned int idx = t + blockDim.x * blockIdx.x;
    source_t source = source_d[idx];
    source_t est_error = (source - mean_d[idx]);
    bool isnan_site = isnan(dilution_mask_d[idx]);

    volatile __shared__ source_t sum_err[256];

    // AVERAGE ABSOLUTE ERROR
    sum_err[t] = isnan_site * fabs(est_error);
    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) AAE_d[blockIdx.x] = sum_err[0];

    // AVERAGE RELAITVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * est_error / source;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) ARE_d[blockIdx.x] = sum_err[0];

    // AVERAGE ABSOLUTE RELATIVE ERROR
    __syncthreads();
    sum_err[t] = isnan_site * fabs(est_error) / source;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) AARE_d[blockIdx.x] = sum_err[0];

    // summation for ROOT AVERAGE SQUARED ROOT
    __syncthreads();
    sum_err[t] = isnan_site * est_error * est_error;

    // unrolling for loop -> to remove instrunction overhead
    __syncthreads();
    if (t < 128) sum_err[t] += sum_err[t + 128];
    __syncthreads();
    if (t < 64) sum_err[t] += sum_err[t + 64];
    __syncthreads();
    if (t < 32)
    {
        sum_err[t] += sum_err[t + 32];
        sum_err[t] += sum_err[t + 16];
        sum_err[t] += sum_err[t + 8];
        sum_err[t] += sum_err[t + 4];
        sum_err[t] += sum_err[t + 2];
        sum_err[t] += sum_err[t + 1];
    }
    if (t == 0) RASE_d[blockIdx.x] = sum_err[0];
}

__global__ void find_temperature_gpu(energy_t* E_source, double* T_ref, double* E_ref, energy_t* E_result, int size_source, int size_refs)
{
    for (int myIdx = blockIdx.x * blockDim.x + threadIdx.x; myIdx < size_source; myIdx += blockDim.x * gridDim.x)
    {
        energy_t myEnergy = E_source[myIdx];
        int it_E = 0;
        int it_T = 0;

        while ((it_E != size_refs - 1) && (myEnergy < E_ref[it_E]))
        {
            ++it_E;
            ++it_T;
        }
        // linear interpolation
        E_result[myIdx] = (it_E == 0) ? (T_ref[it_T]) : ((T_ref[it_T] - T_ref[it_T - 1]) * (myEnergy - E_ref[it_E]) / (E_ref[it_E] - E_ref[it_E - 1]) + T_ref[it_T]);
        //E_result[myIdx] = 0.000001;
        //printf("v blocku %d teplota je %1.4f \t energia je %1.4f\n", myIdx, E_result[myIdx], myEnergy);
    }
}

__global__ void correctTemps(energy_t *T_diluted_per_block_d, energy_t medianValue)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < BLOCKS * 2)
    {
        if (T_diluted_per_block_d[idx] < 0 || T_diluted_per_block_d[idx] > 900.0)
            T_diluted_per_block_d[idx] = medianValue;

    }
}

__global__ void smoothTemps(energy_t *T_diluted_per_block_d, energy_t *spinTemps_raw_d, energy_t *spinTemps_smooth_d, bool first_iteration)
{
    int x = threadIdx.x + BLOCKL*blockIdx.x;
    int y = threadIdx.y + BLOCKL*blockIdx.y;
    int idx = x + L*y;
    if (first_iteration)
    {
        spinTemps_raw_d[idx] = T_diluted_per_block_d[blockIdx.x + gridDim.x*blockIdx.y];
        //spinTemps_smooth_d[idx] = T_diluted_per_block_d[blockIdx.x + gridDim.x*blockIdx.y];
        __syncthreads();
        spinTemps_smooth_d[idx] = spinTemps_raw_d[idx];
        return;
    }
    int max_offset = 8;
    int x_offset = 0;
    int y_offset = 0;
    energy_t newLocalTemp = (energy_t) 0.0;
    int neighbor_count = 0;
    //int debug_index = 515;

    while ((y < L - y_offset) && (y_offset <= max_offset))
    {
        
        while ((x - x_offset) >= 0 && (x_offset <= max_offset))
        {
            //if(idx == debug_index)
                //printf("susedov %d na indexe %d\n", neighbor_count+1, x - x_offset + L*(y + y_offset));
            newLocalTemp += spinTemps_raw_d[x - x_offset + L*(y + y_offset)];
            neighbor_count++;
            x_offset++;
        }
        x_offset = 1;
        
        while (x < L - x_offset && x_offset <= max_offset)
        {
            //if (idx == debug_index)
                //printf("susedov %d na indexe %d\n", neighbor_count+1, x + x_offset + L*(y + y_offset));
            newLocalTemp += spinTemps_raw_d[x + x_offset + L*(y + y_offset)];
            neighbor_count++;
            x_offset++;
        }
        x_offset = 0;
        
        y_offset++;
    }
    y_offset = 1;
    x_offset = 0;
   
    while (y - y_offset >= 0 && y_offset <= max_offset)
    {  
        while (x - x_offset > 0 && x_offset <= max_offset)
        {
            //if (idx == debug_index)
                //printf("susedov %d na indexe %d\n", neighbor_count+1, x - x_offset + L*(y - y_offset));
            newLocalTemp += spinTemps_raw_d[x - x_offset + L*(y - y_offset)];
            neighbor_count++;
            x_offset++;
        }
        x_offset = 1;
        while (x < L - x_offset && x_offset <= max_offset)
        {
            //if (idx == debug_index)
                //printf("susedov %d na indexe %d\n", neighbor_count + 1, x + x_offset + L*(y - y_offset));
            newLocalTemp += spinTemps_raw_d[x + x_offset + L*(y - y_offset)];
            neighbor_count++;
            x_offset++;
        }
        x_offset = 0;
        y_offset++;
    }
    //if(y == 1)
        //printf("thread %d, x = %d a y = %d ma %d susedov\n", idx, x, y, neighbor_count);
    spinTemps_smooth_d[idx] = newLocalTemp / (energy_t)neighbor_count;
}

std::vector<energy_t> smoothTempsCPU(energy_t *spinTemps_d)
{
    std::vector<energy_t> newTemps;
    newTemps.resize(L*L);
    for (unsigned int x = 0; x < L; x++)
    {
        for (unsigned int y = 0; y < L; y++)
        {
            unsigned int idx = x + L*y;
            energy_t newLocalTemp = spinTemps_d[idx];
            unsigned int neighbor_count = 1;
            if (x < L - 1)
            {
                newLocalTemp += spinTemps_d[x + 1 + L*y];
                neighbor_count++;
            }
            if (y < L - 1)
            {
                newLocalTemp += spinTemps_d[x + L*(y + 1)];
                neighbor_count++;
                if (x > 0)
                {
                    newLocalTemp += spinTemps_d[x - 1 + L*(y + 1)];
                    neighbor_count++;
                }
                if (x < L - 1)
                {
                    newLocalTemp += spinTemps_d[x + 1 + L*(y + 1)];
                    neighbor_count++;
                }
            }
            if (x > 0)
            {
                newLocalTemp += spinTemps_d[x - 1 + L*y];
                neighbor_count++;
            }
            if (y > 0)
            {
                newLocalTemp += spinTemps_d[x + L*(y - 1)];
                neighbor_count++;
                if (x > 0)
                {
                    newLocalTemp += spinTemps_d[x - 1 + L*(y - 1)];
                    neighbor_count++;
                }
                if (x < L - 1)
                {
                    newLocalTemp += spinTemps_d[x + 1 + L*(y - 1)];
                    neighbor_count++;
                }
            }
            newTemps[idx] = newLocalTemp / (energy_t)neighbor_count;
        }

    }
    return newTemps;

}

energy_t cpu_energy(spin_t *s)
{
    // double ie = 0;
    energy_t partE = 0;
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
#ifdef DOUBLE_PRECISION
            if (x < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
            if (y < L - 1) partE -= cos(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#else
            if (x < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + 1 + L*y]));
            if (y < L - 1) partE -= cosf(Qfactor * (s[x + L*y] - s[x + L*(y + 1)]));
#endif
        }
    }
    return partE / Nbond;
}

double find_temperature(energy_t E_source, std::vector<double> T_ref, std::vector<double> E_ref)
{
    auto it_E = E_ref.begin();
    auto it_T = T_ref.begin();

    while ((it_E != E_ref.end()) && (E_source < *it_E))
    {
        ++it_E;
        ++it_T;
    }
    // linear interpolation
    return (it_E == E_ref.begin()) ? (*it_T) : ((*it_T - *(it_T - 1)) * (E_source - *it_E) / (*it_E - *(it_E - 1)) + *it_T);
}

// templates
template <class T> T sumPartialSums(T *parSums_d, int length)
{
    std::vector<T> parSums(length);
    CUDAErrChk(cudaMemcpy(parSums.data(), parSums_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    T sum = 0;
    for (auto i : parSums) sum += i;
    return sum;
}

template <class T> std::vector<T> findMinMax(T *min_d, T *max_d, int length)
{
    std::vector<T> min_h(length);
    std::vector<T> max_h(length);
    CUDAErrChk(cudaMemcpy(min_h.data(), min_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    CUDAErrChk(cudaMemcpy(max_h.data(), max_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    /*T min_temp = *(std::min_element(min_h.begin(), min_h.end()));
    T max_temp = *(std::max_element(max_h.begin(), max_h.end()));
    std::vector<T> min_max = { min_temp, max_temp };*/
    std::vector<T> min_max = { min_h.at(0), max_h.at(0) };
    for (auto i : min_h) min_max.at(0) = std::fmin(min_max.at(0), i);
    for (auto i : max_h) min_max.at(1) = std::fmax(min_max.at(1), i);

    /* std::cout << "Block Minimum elements: ";
    for (auto i : min_h) std::cout << i << " ";
    std::cout << "\n"; */

    return min_max;
}

template <class T> T find_median(T *data_d, int length)
{
    std::vector<T> data_h(length);
    CUDAErrChk(cudaMemcpy(data_h.data(), data_d, length * sizeof(T), cudaMemcpyDeviceToHost));
    // First we sort the array 
    std::sort(data_h.begin(), data_h.end());
    // check for even case 
    if (length % 2 != 0)
        return data_h[length / 2];
    return (data_h[(length - 1) / 2] + data_h[length / 2]) / (T)2.0;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(energy_t *d_in, energy_t *d_out)
{
    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    typedef cub::BlockLoad<
        energy_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    //typedef BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;

    typedef cub::BlockStore<
        energy_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<
        energy_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage       load;
        typename BlockStoreT::TempStorage      store;
        typename BlockRadixSortT::TempStorage  sort;
    } temp_storage;
    // Obtain this block's segment of consecutive keys (blocked across threads)
    energy_t thread_keys[ITEMS_PER_THREAD];
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);

    __syncthreads();    // Barrier for smem reuse
                        // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);
    __syncthreads();    // Barrier for smem reuse
                        // Store the sorted segment 
    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
}


// cuRAND errors
char* curandGetErrorString(curandStatus_t rc)
{
    switch (rc) {
    case CURAND_STATUS_SUCCESS:                   return (char*)curanderr[0];
    case CURAND_STATUS_VERSION_MISMATCH:          return (char*)curanderr[1];
    case CURAND_STATUS_NOT_INITIALIZED:           return (char*)curanderr[2];
    case CURAND_STATUS_ALLOCATION_FAILED:         return (char*)curanderr[3];
    case CURAND_STATUS_TYPE_ERROR:                return (char*)curanderr[4];
    case CURAND_STATUS_OUT_OF_RANGE:              return (char*)curanderr[5];
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:       return (char*)curanderr[6];
#if CUDART_VERSION >= 4010 
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return (char*)curanderr[7];
#endif
    case CURAND_STATUS_LAUNCH_FAILURE:            return (char*)curanderr[8];
    case CURAND_STATUS_PREEXISTING_FAILURE:       return (char*)curanderr[9];
    case CURAND_STATUS_INITIALIZATION_FAILED:     return (char*)curanderr[10];
    case CURAND_STATUS_ARCH_MISMATCH:             return (char*)curanderr[11];
    case CURAND_STATUS_INTERNAL_ERROR:            return (char*)curanderr[12];
    default:                                      return (char*)curanderr[13];
    }
}