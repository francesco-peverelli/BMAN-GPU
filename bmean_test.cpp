#include "bmean_test.h"
#include <iostream>
#include <fstream>
#include <map>

#define DEBUG 0

constexpr int MAX_L = 8;
constexpr int MIN_L = 2;
constexpr int MIN_N = 1;
constexpr int MAX_N = 31;
constexpr int MAX_MSA = 150;
constexpr uint32_t K = 9;
constexpr double EDGE_SOLIDITY = 0.0;
constexpr unsigned SOLID_THRESH = 4; 
constexpr unsigned MIN_ANCHORS = 2;
const string SC_PATH = "./BOA/blosum80.mat"; 
const string ERR_FILE_PATH = "./error_file_archive.txt";
const string W_START = "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";
const string W_END   = ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";


static map<int,char> char_map = { { 0, 'A' }, { 1, 'C' }, { 2, 'G' }, { 3, 'T' } };

vector<vector<string>> read_batch(string &filepath, int max_N, int max_W, int batch_size){
	
	ifstream file;
	file.open(filepath);
	if(file.fail()){
		throw ios_base::failure(strerror(errno));
	}

	file.exceptions(file.exceptions() | ios::failbit | ifstream::badbit);
	
	string line;
	unsigned long long collected_samples = 0;
	vector<vector<string>> batch;
	vector<string> current_window;
	int step = 150000;
	bool printed = false;
	unsigned long long lineno = 0;
	while(getline(file, line)){
		lineno++;
		if(lineno >= 28378903) cout << "(" << lineno << "):" << line << endl;
		if(collected_samples >= batch_size){
			cout << "Lines read: " << lineno << endl;
			break;
		}

		if(line[0] == 'A' || line[0] == 'T' || line[0] == 'G' || line[0] == 'C'){
			current_window.push_back(line);

		}else{

			bool window_ok = true;
			if(current_window.size() > max_W)
				window_ok = false;

			int max_size = 0;
			for(auto s : current_window){
				if(s.size() > max_N | !window_ok){
					window_ok = false;
					break;
				}else{
					if(max_size < s.size())
						max_size = s.size();
				}
			}
			if(max_size < max_N / 2){ //we assume to distribute sequences with a doubling rule
				window_ok = false;
			}
			if(window_ok){
				batch.push_back(current_window);
				collected_samples++;
				printed = false;
			}
			current_window.clear();
		}
		if(collected_samples % step == 0 && !printed){
			cout << "Collected samples: " << collected_samples << "/" << batch_size << ", line " << lineno << endl;
			printed = true;
		}
	}
	cout << "out of loop" << endl;
	if(file.bad()){
		throw ios_base::failure(strerror(errno));
	}

	file.exceptions(file.exceptions() | ios::failbit | ifstream::badbit);
	
	file.close();

	cout << "*** SAMPLE COLLECTION COMPLETE ***" << endl;
	cout << "Collected (" << collected_samples << "/" << batch_size << ") samples from reads set" << endl;
	cout << "Max window length = " << max_W << ", max sequence length = " << max_N << endl;
	return batch;
}

vector<string> test_bmean(vector<string> &W, int maxMSA, string path) {

	//sw POA lib call
	//vector<string> result_POA = consensus_POA( W, maxMSA, path );

	//return result_POA;
	cout << "ERROR:: This is not a function!!!!!\n";
	return vector<string>();
}

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC(vector<vector<string>> &test_in){

	vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> res;
	res.reserve(BATCH_SIZE);

	for(vector<string> &W : test_in){
		res.push_back(MSABMAAC(W, K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, SC_PATH));
	}
	return res;
}

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_gpu(vector<vector<string>> &test_in){

	int pool_size = 2;
	int jobs_to_load = test_in.size();
	ctpl::thread_pool my_pool(pool_size);
	vector< std::future< pair< vector< poa_gpu_utils::Task<vector<string> > >, unordered_map<kmer, unsigned> > > > enq_res(jobs_to_load);
	vector<vector<poa_gpu_utils::Task<vector<string>>>> sched_tasks(jobs_to_load);
	vector< std::future< vector<vector<string>> > > deq_res(jobs_to_load);
	vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> result(jobs_to_load);

	//cout << "[GPU-TEST]: init gpu test\n";

	MSABMAAC_gpu_init_ctpl(BATCH_SIZE, my_pool);
	pool_size--; //account for the executor thread

	//cout << "[GPU-TEST]: init gpu done\n";

	//load the first jobs
	int next_job = 0;
	while(next_job < pool_size && next_job < jobs_to_load){
		//cout << "[GPU-TEST]: loading job " << next_job << "\n";
		enq_res[next_job] = my_pool.push(MSABMAAC_gpu_enqueue_ctpl,
					test_in[next_job], K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, SC_PATH);
		next_job++;
	}
	pair< vector< poa_gpu_utils::Task<vector<string> > >, unordered_map<kmer, unsigned>> f_res;
	int curr_job = 0;

	cout << "[GPU-TEST]: progressive enqueue...\n";

	while(next_job < jobs_to_load){
	
		//cout << "[GPU-TEST]: getting job " << curr_job << "\n";
		
		enq_res[curr_job].wait();

		//cout << "[GPU-TEST]: job " << curr_job << " is ready\n";

		f_res = enq_res[curr_job].get();
		//cout << "[GPU-TEST]: get done\n";
		result[curr_job].second = f_res.second; 
		//cout << "[GPU-TEST]: =1 done\n"; 
		sched_tasks[curr_job] = f_res.first;
		for(auto a : sched_tasks[curr_job])
		cout << "T ID=" << a.task_id << "\n";
		//cout << "[GPU-TEST]: =2 done\n"; 
		curr_job++;
		//cout << "[GPU-TEST]: loading job " << next_job << "\n";
		enq_res[next_job] = my_pool.push(MSABMAAC_gpu_enqueue_ctpl,
					test_in[next_job], K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, SC_PATH);
		next_job++;
			
	}	

	//cout << "[GPU-TEST]: enqueue trail...\n";

	//wait until the remaining jobs are enqueued
	while(curr_job < jobs_to_load){
		f_res = enq_res[curr_job].get();
		result[curr_job].second = f_res.second;
		sched_tasks[curr_job] = f_res.first; 
		for(auto a : sched_tasks[curr_job])
		cout << "T ID=" << a.task_id << "\n";
		curr_job++;
	}

	//cout << "[GPU-TEST]: starting flush...\n";

	//execute all remaining POA tasks 
	MSABMAAC_gpu_flush();

	//cout << "[GPU-TEST]: dequeue start\n";

	//load the first dequeue jobs
	next_job = 0;
	while(next_job < pool_size && next_job < jobs_to_load){
		//cout << "[GPU-TEST]: loading job " << next_job << "\n";
		deq_res[next_job] = my_pool.push(MSABMAAC_gpu_dequeue_ctpl, sched_tasks[next_job]);
		next_job++;
	}

	//cout << "[GPU-TEST]: progressive dequeue...\n";

	curr_job = 0;
	while(next_job < jobs_to_load){
		
		result[curr_job].first = deq_res[curr_job].get();
		curr_job++;
		//cout << "[GPU-TEST]: loading job " << next_job << "\n";
		deq_res[next_job] = my_pool.push(MSABMAAC_gpu_dequeue_ctpl, sched_tasks[next_job]);
		next_job++;		
	}

	//cout << "[GPU-TEST]: dequeue trail...\n";

	while(curr_job < jobs_to_load){
		result[curr_job].first = deq_res[curr_job].get();
		curr_job++;
	}
	MSABMAAC_gpu_done();
	cout << "MSA test ALL DONE\n";

	return result;
}

vector<vector<string>> convert_MSA_batch(vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &in){
	
	vector<vector<string>> res(in.size());
	int i = 0;
	for(pair<vector<vector<string>>, unordered_map<kmer, unsigned>> &e : in){
		if(e.first.size() == 0){
			cout << "Empty res\n";
			res[i] = vector<string>();
			continue;
		}

		res[i] = (e.first)[0];
		i++;
	}
	return res;
}

void test_MSA_batch(vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &obtained, 
		    vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &expected){

	auto c_obt = convert_MSA_batch(obtained);
	auto c_exp = convert_MSA_batch(expected);

	cout << "Conversion done\n";

	test_batch(c_obt, c_exp);
}

vector<string> generate_random_window(int max_L, int min_L, int min_N, int max_N) {

	random_device rd;
	default_random_engine generator(rd());
	
	uniform_int_distribution<int> L_distribution(min_L, max_L);
	uniform_int_distribution<int> N_distribution(min_N, max_N);	
	uniform_int_distribution<int> char_distribution(0,3);

	int L = L_distribution(generator);
	
	vector<string> window;

	for(int i = 0; i < L; i++){

		int N = N_distribution(generator);
		string sequence = "";
	
		for(int j = 0; j < N; j++){
			char c = char_map[char_distribution(generator)];
			sequence += c;
			
		}
		window.push_back(sequence);
	}
	return window;
}

void get_bmean_batch_result(vector<vector<string>> windows, vector<vector<string>> &results) {	
	
	//omp_set_dynamic(0);
	//omp_set_num_threads(4);
	//cout << "Executing " << omp_get_num_threads() << " threads" << endl; 
	
	for(int i = 0; i <  windows.size(); i++){
		vector<string> result = test_bmean(windows[i],MAX_MSA, SC_PATH);
		results.push_back(result);
	}
}

void write_window(ofstream &file, vector<string> window_o, vector<string> window_e) {

	file << W_START << endl;
	file << "<<obtained>>" << endl;
	for(auto sequence : window_o){
		file << sequence << endl;
	}
	file << endl;
	file << "<<expected>>" << endl; 
	for(auto sequence : window_e){
		file << sequence << endl;
	}

	file << W_END << endl;
}

void test_task_batch(vector<poa_gpu_utils::Task<vector<string>>> &obtained, vector<vector<string>> &expected) {
	
	ofstream err_file;
	int b_size = obtained.size();
	int correct = b_size;
	
	bool window_correct;
	bool all_correct = true;
	
	err_file.open(ERR_FILE_PATH, ios::out | ios::app );
	if(err_file.fail()){
		throw ios_base::failure(strerror(errno));
	}

	err_file.exceptions(err_file.exceptions() | ios::failbit | ifstream::badbit);
	
	if(obtained.size() != expected.size()){
		cout << "Unexpected batch size difference: " << obtained.size() << " expected: " << expected.size() << endl;
	}	

	for(int i = 0; i < obtained.size(); i++){
		
		window_correct = true;		
#if DEBUG
		cout << "Testing window " << i + 1 << endl;
		cout << "Task id = " << obtained[i].task_id << endl;
#endif		
		vector<string> &window_o = obtained[i].task_data;
		vector<string> &window_e = expected[obtained[i].task_id];

		if(window_e.size() != window_o.size()){
			cout << "Unmatching window size" << endl;
			write_window(err_file, window_o, window_e);
			window_correct = false;
			all_correct = false;
			correct--;
			continue;
		}
		
		for(int j = 0; j < window_o.size(); j++){
			
			if(!window_correct)
				break;

			string str_o = window_o[j];
			string str_e = window_e[j];
		
			if(str_o.size() != str_e.size()){
				cout << "\tUnmatching sequence length: " << str_o.size() << ", expected " << str_e.size() << endl;
				write_window(err_file, window_o, window_e); 
				window_correct = false;
				all_correct = false;
				correct--;
				continue;
			}
			
			for(int k = 0; k < str_o.size(); k++) {
				if(!window_correct)
					break;
				if( str_o[k] != str_e[k] ){ 
					cout << "\tcharacter mismatch" << endl;
					write_window(err_file, window_o, window_e); 
					window_correct = false;
					all_correct = false;
					correct--;
					continue;
				}
			}
		}
#if DEBUG		
		if(window_correct){
			cout << "WINDOW_OK" << endl;
		}
#endif
	}
	
	if(all_correct){
		cout << "BATCH_PASSED" << endl;
	}else{
		cout << "PASSED= " << correct << "/" << b_size << endl;
	}
	
	err_file.close();
}
void test_batch(vector<vector<string>> &obtained, vector<vector<string>> &expected) {
	
	ofstream err_file;
	int b_size = obtained.size();
	int correct = b_size;
	
	bool window_correct;
	bool all_correct = true;
	
	err_file.open(ERR_FILE_PATH, ios::out | ios::app );
	if(err_file.fail()){
		throw ios_base::failure(strerror(errno));
	}

	err_file.exceptions(err_file.exceptions() | ios::failbit | ifstream::badbit);
	
	if(obtained.size() != expected.size()){
		cout << "Unexpected batch size difference: " << obtained.size() << " expected: " << expected.size() << endl;
	}	

	for(int i = 0; i < obtained.size(); i++){
		
		window_correct = true;		
#if DEBUG
		cout << "Testing window " << i + 1 << endl;
#endif		
		vector<string> window_o = obtained[i];
		vector<string> window_e = expected[i];

		if(window_e.size() != window_o.size()){
			cout << "Unmatching window size" << endl;
			write_window(err_file, window_o, window_e);
			window_correct = false;
			all_correct = false;
			correct--;
			continue;
		}
		
		for(int j = 0; j < window_o.size(); j++){
			
			if(!window_correct)
				break;

			string str_o = window_o[j];
			string str_e = window_e[j];
		
			if(str_o.size() != str_e.size()){
				cout << "\tUnmatching sequence length: " << str_o.size() << ", expected " << str_e.size() << endl;
				write_window(err_file, window_o, window_e); 
				window_correct = false;
				all_correct = false;
				correct--;
				continue;
			}
			
			for(int k = 0; k < str_o.size(); k++) {
				if(!window_correct)
					break;
				if( str_o[k] != str_e[k] ){ 
					cout << "\tcharacter mismatch" << endl;
					write_window(err_file, window_o, window_e); 
					window_correct = false;
					all_correct = false;
					correct--;
					continue;
				}
			}
		}
#if DEBUG		
		if(window_correct){
			cout << "WINDOW_OK" << endl;
		}
#endif
	}
	
	if(all_correct){
		cout << "BATCH_PASSED" << endl;
	}else{
		cout << "PASSED= " << correct << "/" << b_size << endl;
	}
	
	err_file.close();
}

vector<vector<string>> get_random_sample(int batch_size, int max_L = MAX_L, int min_L = MIN_L, int max_N = MAX_N, int min_N = MIN_N) { 
	
	cout << "Alive!\n";
	vector<vector<string>> sample;	
	int step = batch_size / 10;
	int perc = 0;

	cout << "Sample generation: size=" << batch_size << ", L=[" << min_L << "," << max_L << "], N=[" << min_N << "," << max_N << "]\n";


	for(int i = 0; i < batch_size; i++) {
		sample.push_back(generate_random_window(max_L, min_L, min_N, max_N));
		if(i % step == 0){ cout << "Generation: [" << perc << "%]\n"; perc += 10;  }
	}
	return sample;
}


