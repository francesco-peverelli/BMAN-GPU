#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <numeric>
#include <stdexcept>
#include <condition_variable>
#include "bmean_test.h"

#define TEST 1

using namespace std;
using namespace chrono;

#define NOW high_resolution_clock::now() 

//poa_gpu_utils::TaskConcurrencyManager CM(NUM_TASK_TYPES, BATCH_SIZE);

int check_input_int(string &arg){
	
	try{
		size_t pos;
		int arg_i = stoi(arg, &pos);
		if(pos < arg.size()){
			std::cerr << "Trailing characters after number: " << arg << '\n';
		}
		return arg_i;
	} catch (invalid_argument const &ex) {
		std::cerr << "Invalid number: " << arg << '\n';
		return -1;
	} catch (out_of_range const &ex) {
		std::cerr << "Number out of range: " << arg << '\n';
		return -1;
	}
	
}

int main(int argc, char* argv[]) {

	bool read_from_file = false;
	char* path_ref;	
	
	if(argc < 4){
		cout << "Invalid arguments. Call this program as: ./testGPU maxSeqSize maxWindowSize sampleSize [read_file.pow]" << endl;
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
		input = read_batch(filepath, SEQ_LEN, WLEN, N_ALIGNMENTS);
	}else{
		cout << "*** GENERATING RANDOM SAMPLE OF SIZE " << N_ALIGNMENTS << " ***" << endl;
		input = get_random_sample(N_ALIGNMENTS, WLEN, MIN_WLEN, SEQ_LEN, MIN_SLEN);
	}

	vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> result_BMEAN;
	vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> result_GPU;

#if TEST
	cout << "CPU start" << endl;
	
	auto CPU_start = NOW;

	result_BMEAN = testMSABMAAC_pool(input); 

	cout << "CPU done" << endl;
#endif
	auto CPU_end = NOW;

	cout << "GPU start\n";
	
	result_GPU = testMSABMAAC_gpu_std(input);
	
	auto GPU_end = NOW;
	
#if TEST
	cout << "Test start" << endl;
	test_MSA_batch(result_GPU, result_BMEAN);

	auto duration_CPU = duration_cast<microseconds>(CPU_end - CPU_start);
	cout << "CPU time = " << duration_CPU.count() << " microseconds" << endl;
#endif
	auto duration_GPU = duration_cast<microseconds>(GPU_end - CPU_end);
	cout << "GPU time = " << duration_GPU.count() << " microseconds" << endl;
#if TEST
	cout << "Speedup = " << (double)duration_CPU.count() / (double)duration_GPU.count() << endl;
#endif	
	return 0;
}
