#ifndef BMEAN_TEST_H
#define BMEAN_TEST_H

#include "utils.h"
#include "bmean.h"
#include <unordered_map>
#include <vector>
#include <random>
#include "global_tasks_manager.h"


using namespace std;

void read_batch_2(vector<vector<string>> &batch, size_t size, string filename);

vector<vector<string>> read_batch(string &filepath, int max_N, int max_W, int batch_size);

void get_bmean_batch_result(vector<vector<string>> windows, vector<vector<string>> &results);

vector<string> test_bmean(vector<string> &W, int maxMSA, string path);

void test_MSA_batch(vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &obtained, 
		    vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &expected);

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC(vector<vector<string>> &test_in);

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_pool(vector<vector<string>> &test_in);

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_gpu(vector<vector<string>> &test_in);
	
vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_gpu_std(vector<vector<string>> &test_in);

vector<string> generate_random_window(int max_L, int min_L, int max_N);

void get_bmean_batch_result_mt(vector<vector<string>> &windows, vector<vector<string>> &results, unsigned int n_threads);

void get_bmean_batch_result_gpu(vector<vector<string>> windows, vector<vector<string>> &results, int &c, int max_s, int max_w);

void test_batch(vector<vector<string>> &obtained, vector<vector<string>> &expected);

void test_task_batch(vector< poa_gpu_utils::Task< vector< string > > > &obtained, vector<vector<string>> &expected);

vector<vector<string>> get_random_sample(int batch_size, int max_L, int min_L, int max_N, int min_N);
	
#endif /*BMEAN_TEST_H*/

