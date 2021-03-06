#include "bmean_test.h"
#include "poa-gpu.h"
#include "tasks_manager.h"
#include <iostream>
#include <fstream>
#include <map>

using namespace std::chrono;

#define DEBUG 0
#define NOW high_resolution_clock::now()

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
//constexpr int N_THREADS = 64;

static map<int,char> char_map = { { 0, 'A' }, { 1, 'C' }, { 2, 'G' }, { 3, 'T' } };

vector<string> test_bmean(vector<string> &W, int maxMSA, string path) {

	//sw POA lib call
	vector<string> result_POA = consensus_POA( W, maxMSA, path );

	return result_POA;

}

vector<vector<string>> read_batch(string &filepath, int max_N, int max_W, int batch_size){
	
	string line;
	unsigned long long collected_samples = 0;
	vector<vector<string>> batch;
	vector<string> current_window;
	int step = 150000;
	bool printed = false;
	unsigned long long lineno = 0;

	for(int i = 0; i < 50; i++){
	
	ifstream file(filepath);

	while(getline(file, line)){
		lineno++;
		if(collected_samples >= batch_size){
			cout << "Lines read: " << lineno << endl;
			break;
		}

		if(line[0] == 'A' || line[0] == 'T' || line[0] == 'G' || line[0] == 'C'){
			current_window.push_back(line);

		}else{

			bool window_ok = true;
			/*if(current_window.size() > max_W)
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
			}*/
			if(window_ok){
				batch.push_back(current_window);
				collected_samples++;
				printed = false;
			}
			vector<string>().swap(current_window);
		}
		if(collected_samples % step == 0 && !printed){
			cout << "Collected samples: " << collected_samples << "/" << batch_size << ", line " << lineno << endl;
			printed = true;
		}
	}
	cout << "out of loop" << endl;
	file.close();
	}

	cout << "*** SAMPLE COLLECTION COMPLETE ***" << endl;
	cout << "Collected (" << collected_samples << "/" << batch_size << ") samples from reads set" << endl;
	cout << "Max window length = " << max_W << ", max sequence length = " << max_N << endl;
	return batch;
}

void read_batch_2(vector<vector<string>> &batch, size_t size, string filename){

    std::ifstream infile(filename);
    std::string line;
    int n = 0;
    int i = 0;
    while (getline(infile, line))
    {
        if (n == 0)
        {
            n = stoi(line);
            batch.emplace_back(std::vector<std::string>());
        }
        else
        {
            batch.back().push_back(line);
            n--;
        }
	i++;
    }

}

void execute_bmean(vector<vector<string>> batch, vector<vector<string>> &batch_result){
	
	int size = batch.size();	
	for(int i = 0; i < size; i++){
		batch_result[i] = test_bmean(batch[i], MAX_MSA, SC_PATH);
	}
}

void get_bmean_batch_result_mt(vector<vector<string>> &windows, vector<vector<string>> &results, unsigned int n_threads) {	
	
	cout << "n threads: " << n_threads << endl;
	int size = windows.size();
	int batch_size = size/n_threads;
	thread threads[n_threads];
	vector<vector<vector<string>>> batches(n_threads);
	vector<vector<vector<string>>> batch_results(n_threads);	

	for(int i = 0; i < n_threads; i++){
		//cout << "thread " << i << ", batch_size=" << batch_size << "\n";
		batches[i] = vector<vector<string>>(batch_size);
		batch_results[i] = vector<vector<string>>(batch_size);
		for(int j = 0; j < batch_size; j++){
			if(i * batch_size + j < windows.size()){
				//cout << "b[" << i << "][" << j << "] --> windows[" << i * batch_size + j << "]\n";
				batches[i][j] = windows[i * batch_size + j];
			}
		}
		//cout << "t[" << i << "exec\n";
		threads[i] = thread(execute_bmean, batches[i], ref(batch_results[i]));
	}

	for(int i = 0; i < n_threads; i++){
		threads[i].join();
	}

	for(int i = 0; i <  size; i++){
		//if((i%N_THREADS) * batch_size + i/N_THREADS < size){
		 	//cout << "res[" << i << "] <-- b_res[" << i/batch_size << "][" << i%batch_size << "]\n";
			results[i] = batch_results[i/batch_size][i%batch_size];
		//}
	}
	//exit(0);
}


void get_bmean_batch_result_gpu(vector<vector<string>> windows, vector<vector<string>> &results, int &c, int max_s, int max_w){

	poa_gpu_utils::TaskRefs T;
	poa_gpu_utils::TaskType TTy;
	size_t size = windows.size();

	vector<vector<string>> result_GPU;
	vector<poa_gpu_utils::Task<vector<string>>> gpu_tasks(size, poa_gpu_utils::Task<vector<string>>(0,0,vector<string>()));
	vector<poa_gpu_utils::Task<vector<string>>> gpu_res(size, poa_gpu_utils::Task<vector<string>>(0,0,vector<string>()));
	
	//task transfer
	int i = 0;
	for(auto s : windows){
		poa_gpu_utils::Task<vector<string>> t = poa_gpu_utils::Task<vector<string>>(i, i, s);
		gpu_tasks[i] = t;
		i++;
	}

	TTy = poa_gpu_utils::get_task_type_direct(max_w, max_s);

	auto start = NOW;

	poa_gpu_utils::sel_gpu_POA_alloc(T, TTy);
	poa_gpu_utils::sel_gpu_POA(gpu_tasks, T, gpu_res, 0, TTy);
	poa_gpu_utils::gpu_POA_free(T,TTy);

	auto end = NOW;
	c = duration_cast<microseconds>(end - start).count();

	for(auto r : gpu_res){
		results.push_back(r.task_data);
	}
}

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC(vector<vector<string>> &test_in){

	vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> res;
	res.reserve(BATCH_SIZE);

	for(vector<string> &W : test_in){
		res.push_back(MSABMAAC(W, K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, SC_PATH));
	}
	return res;
}

pair<vector<vector<string>>, unordered_map<kmer, unsigned>> MSABMAAC_ctpl(int id, const vector<string>& nadine,uint32_t la,double cuisine, unsigned solidThresh, unsigned minAnchors, unsigned maxMSA, string path){
	return  MSABMAAC(nadine,la,cuisine,solidThresh,minAnchors,maxMSA,path);
}

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_pool(vector<vector<string>> &test_in){
	
	int pool_size = 1000;
	int jobs_to_load = test_in.size();
	ctpl::thread_pool my_pool(pool_size);
	vector<future<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>>> res(jobs_to_load);
	vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> result(jobs_to_load);
	int next_job = 0;
	while(next_job < pool_size && next_job < jobs_to_load){
		res[next_job] = my_pool.push(MSABMAAC_ctpl, test_in[next_job], K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, SC_PATH);
		next_job++;
	}
	int curr_job = 0;
	while(next_job < jobs_to_load){
		result[curr_job] = res[curr_job].get();
		curr_job++;
		res[next_job] = my_pool.push(MSABMAAC_ctpl, test_in[next_job], K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, SC_PATH);
		next_job++;
	}

	while(curr_job < jobs_to_load){
		result[curr_job] = res[curr_job].get();
		curr_job++;
	}
	return result;
}

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_gpu(vector<vector<string>> &test_in){

	int pool_size = 1000;
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

	//cout << "[GPU-TEST]: progressive enqueue...\n";

	while(next_job < jobs_to_load){
	
		//cout << "[GPU-TEST]: getting job " << curr_job << "\n";
		
		enq_res[curr_job].wait();

		//cout << "[GPU-TEST]: job " << curr_job << " is ready\n";

		f_res = enq_res[curr_job].get();
		//cout << "[GPU-TEST]: get done\n";
		result[curr_job].second = f_res.second; 
		sched_tasks[curr_job] = f_res.first;
		curr_job++;
		//cout << "[GPU-TEST]: loading job " << next_job << "\n";
		enq_res[next_job] = my_pool.push(MSABMAAC_gpu_enqueue_ctpl,
					test_in[next_job], K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, SC_PATH);
		next_job++;
			
	}	

	cout << "[GPU-TEST]: enqueue trail...\n";

	//wait until the remaining jobs are enqueued
	while(curr_job < jobs_to_load){
		f_res = enq_res[curr_job].get();
		result[curr_job].second = f_res.second;
		sched_tasks[curr_job] = f_res.first; 
		curr_job++;
	}

	cout << "[GPU-TEST]: starting flush...\n";

	//execute all remaining POA tasks 
	MSABMAAC_gpu_flush();

	cout << "[GPU-TEST]: dequeue start\n";

	//load the first dequeue jobs
	next_job = 0;
	while(next_job < pool_size && next_job < jobs_to_load){
		//cout << "[GPU-TEST]: loading job " << next_job << "\n";
		deq_res[next_job] = my_pool.push(MSABMAAC_gpu_dequeue_ctpl, sched_tasks[next_job], MAX_MSA, SC_PATH);
		next_job++;
	}

	cout << "[GPU-TEST]: progressive dequeue...\n";

	curr_job = 0;
	while(next_job < jobs_to_load){
		
		//deq_res[curr_job].wait();
		result[curr_job].first = deq_res[curr_job].get();
		curr_job++;
		//cout << "[GPU-TEST]: loading job " << next_job << "\n";
		deq_res[next_job] = my_pool.push(MSABMAAC_gpu_dequeue_ctpl, sched_tasks[next_job], MAX_MSA, SC_PATH);
		next_job++;		
	}

	cout << "[GPU-TEST]: dequeue trail...\n";

	while(curr_job < jobs_to_load){
		result[curr_job].first = deq_res[curr_job].get();
		curr_job++;
	}
	MSABMAAC_gpu_done();
	cout << "MSA test ALL DONE\n";

	return result;
}

void MSABMAAC_gpu_dequeue_batch(vector<vector<poa_gpu_utils::Task<vector<string>>>> &sched_tasks, vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &result, vector<bool> &needs_enqueue, int n_threads, int tid, unsigned maxMSA, const string &path){

	int n = sched_tasks.size();
	for(int i = tid; i < n; i+=n_threads){
		if(!needs_enqueue[i]){
			continue;
		}
		result[i].first = MSABMAAC_gpu_dequeue(sched_tasks[i], maxMSA, path);	
	}
}

void MSABMAAC_gpu_enqueue_batch(vector<vector<string>>& input, vector<vector<poa_gpu_utils::Task<vector<string>>>> &sched_tasks, vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &result, vector<bool> &needs_enqueue, int n_threads, int tid, uint32_t la,double cuisine, unsigned solidThresh, unsigned minAnchors, unsigned maxMSA, const string& path){

	int n = input.size();
	for(int i = tid; i < n; i+=n_threads){
		for(auto &V : input){
			if(needs_poa(V)){
				needs_enqueue[i] = true;
				break;
			}
		}
		auto p = MSABMAAC_gpu_enqueue(tid, input[i], la, cuisine, solidThresh, minAnchors, maxMSA, path);
		sched_tasks[i] = p.first;
		result[i].second = p.second;
		
		if(!needs_enqueue[i]){
			vector<vector<string>> res;
			if(input[i].size() != 0){
				res = {{ input[i][0] }};
			}else{
				res = {{ "" }};
			}
			result[i].first = res;
		}
	}
}

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_gpu_std(vector<vector<string>> &test_in){

	int jobs_to_load = test_in.size();
	vector<vector<poa_gpu_utils::Task<vector<string>>>> sched_tasks(jobs_to_load);
	vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> result(jobs_to_load);
	vector<bool> needs_enqueue(jobs_to_load, false);

	//cout << "[GPU-TEST]: init gpu test\n";
	
	//auto init_t = NOW;

	std::thread exec_thread;
	MSABMAAC_gpu_init_batch(BATCH_SIZE, exec_thread);

	int n_threads = 64;
	thread task_threads[64];

	//cout << "[GPU-TEST]: init gpu done\n";
	//auto enq_t = NOW;


	for(int i = 0; i < 64; i++){
		task_threads[i] = std::thread(MSABMAAC_gpu_enqueue_batch, ref(test_in), ref(sched_tasks), ref(result), ref(needs_enqueue), n_threads, i, K, EDGE_SOLIDITY, SOLID_THRESH, MIN_ANCHORS, MAX_MSA, ref(SC_PATH));
	
	}

	for(int i = 0; i < 64; i++){
		task_threads[i].join();
	}

	//cout << "[GPU-TEST]: enqueue done\n";

	//auto flush_t = NOW;

	//execute all remaining POA tasks 
	MSABMAAC_gpu_flush();

	//auto deq_t = NOW;

	for(int i = 0; i < 64; i++){
		task_threads[i] = std::thread(MSABMAAC_gpu_dequeue_batch, ref(sched_tasks), ref(result), ref(needs_enqueue), n_threads, i, (unsigned)MAX_MSA, ref(SC_PATH));
	}

	for(int i = 0; i < 64; i++){
		task_threads[i].join();
	}

	MSABMAAC_gpu_done();

	exec_thread.join();


	//auto end = NOW;

	/*auto init = duration_cast<microseconds>(enq_t - init_t);
	auto enq = duration_cast<microseconds>(flush_t - enq_t);
	auto flush = duration_cast<microseconds>(deq_t - flush_t);
	auto deq = duration_cast<microseconds>(end - deq_t);

	cout << "Init=" << init.count() << endl;
	cout << "Enq=" << enq.count() << endl;
	cout << "Flush=" << flush.count() << endl;
	cout << "Deq=" << deq.count() << endl;*/

	//cout << "MSA test ALL DONE\n";

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
		for(int i = 0; i < 15; i++)
		cout << "BATCH_PASSED" << endl;

	}else{
		for(int i = 0; i < 15; i++) 
		cout << "PASSED= " << correct << "/" << b_size << endl;
		exit(-1);
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
	if(all_correct) cout << "ALL OK" << endl;
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
