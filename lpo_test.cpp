#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <numeric>
#include <stdexcept>
#include "cuda-poa.cuh"
#include "bmean.h"
#include "bmean_test.h"
#include "input_utils.h"

#define TEST 1
#define CPU 0

using namespace std;
using namespace chrono;
using namespace in_utils;

#define NOW high_resolution_clock::now() 

constexpr unsigned int n_threads = 80;

int main(int argc, char* argv[]) {

	bool read_from_file = false;
	char* path_ref;	
	
	if(argc < 4){
		cout << "Invalid arguments. Call this program as: ./poa maxSeqSize maxWindowSize sampleSize [read_file.pow]" << endl;
		return 0;
	}

	if(argc == 5){
		read_from_file = true;
		path_ref = argv[4];
	}
	
	string max_seq_size = argv[1];
	const int SEQ_LEN = check_input_int(max_seq_size);		
	
	string max_w_size = argv[2];
	const int WLEN = check_input_int(max_w_size);		
	
	string sample_size = argv[3];
	const int N_ALIGNMENTS = check_input_int(sample_size);		

	if(SEQ_LEN < 0){
		cout << "Invalid max sequence length provided" << endl;
		return 0;
	}
	if(WLEN < 0){
		cout << "Invalid max window size provided" << endl;
		return 0;
	}
	if(N_ALIGNMENTS < 0){
		cout << "Invalid max window size provided" << endl;
		return 0;
	}
	vector<vector<string>> input;
	
	if(read_from_file){
		string filepath(path_ref);
		cout << "*** ATTEMPTING TO READ FROM " << filepath << " SAMPLE OF SIZE " << N_ALIGNMENTS << " ***" << endl;
		read_batch_2(input, N_ALIGNMENTS, filepath);
		cout << "Read " << input.size() << " alignments" << endl;
	}else{
	//	cout << "*** GENERATING RANDOM SAMPLE OF SIZE " << N_ALIGNMENTS << " ***" << endl;
		input = get_random_sample(N_ALIGNMENTS, WLEN, MIN_WLEN, SEQ_LEN, MIN_SLEN);
	}
		
#if TEST
	cout << "CPU start" << endl;
	
	auto CPU_start = NOW;
	vector<vector<string>> result_CPU(input.size());

	get_bmean_batch_result_mt(input, result_CPU, n_threads);

	cout << "CPU done" << endl;

	auto CPU_end = NOW;
#endif
#if (TEST == 0) 
	auto GPU_start = NOW;

	cout << "[GPU-TEST]: init gpu test\n";
	
	//auto init_t = NOW;

	std::thread exec_thread;
	MSABMAAC_gpu_init_batch(N_ALIGNMENTS, exec_thread);

	cout << "[GPU-TEST]: init gpu done\n";
	//auto enq_t = NOW;

	auto tmp = easy_enqueue(input, N_ALIGNMENTS, 150, "");

	cout << "[GPU-TEST]: enqueue done\n";

	//auto flush_t = NOW;

	//execute all remaining POA tasks 
	MSABMAAC_gpu_flush();

	//auto deq_t = NOW;

	cout << "[GPU-TEST]: flush done\n";

	MSABMAAC_gpu_done();

	exec_thread.join();

	auto result = MSABMAAC_get_gpu_results();

	cout << "[GPU-TEST]: collected result\n";

	MSABMAAC_exit();

	auto GPU_end = NOW;
#endif	
#if TEST
	vector<vector<string>> result_GPU;

	//SIMPLE GPU EXECUTION SINGLE KERNEL
	int c = 0;

	get_bmean_batch_result_gpu(input, result_GPU, c, SEQ_LEN, WLEN);

	cout << "Test start" << endl;

	test_batch(result_GPU, result_CPU);

	auto duration_CPU = duration_cast<microseconds>(CPU_end - CPU_start);
	cout << "CPU time = " << duration_CPU.count() << " microseconds" << endl;
#endif
#if (CPU == 0)	
	//auto duration_GPU = m;//duration_cast<microseconds>(GPU_end - GPU_start);
	cout << "GPU time = " << c << " microseconds" << endl;
#endif
#if TEST
	cout << "Speedup = " << (double)duration_CPU.count() / (double)c << endl;
#endif
	return 0;
}
