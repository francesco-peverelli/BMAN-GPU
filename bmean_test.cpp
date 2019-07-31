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
	vector<string> result_POA = consensus_POA( W, maxMSA, path );

	return result_POA;

}

vector<string> test_global_consensus(int n_windows, vector<vector<string>> &W){

	vector<vector<string>> read_consensus(n_windows);
	int i = 0;
	int j = 0;
	int size = W.size();
	vector<string> results(size/n_windows);
	while(i < size){
		for(vector<string> &el : read_consensus){
			el = W[i]; 
			i++;
		}
		results[j] = global_consensus(read_consensus, read_consensus.size(), MAX_MSA, SC_PATH)[0][0];
		j++;
	}	
	return results;
}

vector<string> test_global_consensus_gpu(int n_windows, vector<vector<string>> &W){
	
	ctpl::thread_pool my_pool(32);
	
	int nty = NUM_TASK_TYPES;
	
	poa_gpu_utils::SyncMultitaskQueues<vector<string>> &q_ref = *(CM.poa_queues);
	my_pool.push(poa_executor_worker, ref(q_ref), CM.task_refs, ref(CM.queue_rdy_mutex), 
		     ref(CM.output_rdy_mutex), ref(CM.queue_rdy_var), ref(CM.output_rdy_var), 
		     CM.exec_notified, CM.flush_mode, CM.processing_required, 
		     CM.current_task, CM.previous_task, 
		     CM.results, nty, CM.current_res_index);
	
	vector<vector<string>> read_consensus(n_windows);
	int i = 0;
	int j = 0;
	int size = W.size();
	vector<string> results(size/n_windows);
	while(i < size){
		for(vector<string> &el : read_consensus){
			el = W[i]; 
			i++;
		}
		vector<vector<string>> V;
		my_pool.push(gpu_global_consensus_worker, V, read_consensus, MAX_MSA, SC_PATH);
		CM.register_preprocessing_task();
		results[j] = V[0][0];
		j++;
	}	
	CM.wait_and_flush_all();	

	return results;
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
	
	for(auto sequence : window_o){
		file << sequence << endl;
	}
	file << endl;
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
void test_batch(vector<vector<string>> obtained, vector<vector<string>> expected) {
	
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
	
	vector<vector<string>> sample;
	for(int i = 0; i < batch_size; i++) {
		sample.push_back(generate_random_window(max_L, min_L, min_N, max_N));
	}
	return sample;
}


