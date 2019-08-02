#ifndef BMEAN_TEST_H
#define BMEAN_TEST_H

#include <unordered_map>
#include <vector>
#include <random>
#include "bmean.h"
#include "global_tasks_manager.h"
#include "../../CTPL/ctpl_stl.h"

using namespace std;

vector<vector<string>> read_batch(string &filepath, int max_N, int max_W, int batch_size);

void get_bmean_batch_result(vector<vector<string>> windows, vector<vector<string>> &results);

vector<string> test_bmean(vector<string> &W, int maxMSA, string path);

//vector<string> test_global_consensus(int n_windows, vector<vector<string>> &W);

//vector<string> test_global_consensus_gpu(int n_windows, vector<vector<string>> &W);

vector<string> generate_random_window(int max_L, int min_L, int max_N);

void test_batch(vector<vector<string>> obtained, vector<vector<string>> expected);

void test_task_batch(vector< poa_gpu_utils::Task< vector< string > > > &obtained, vector<vector<string>> &expected);

vector<vector<string>> get_random_sample(int batch_size, int max_L, int min_L, int max_N, int min_N);
	
#endif /*BMEAN_TEST_H*/

