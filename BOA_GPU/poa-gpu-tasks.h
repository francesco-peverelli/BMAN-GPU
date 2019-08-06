#ifndef POA_GPU_TASKS_H
#define POA_GPU_TASKS_H

#include<map>
#include "poa-constants.h"

#define NUM_TASK_TYPES 9
#define N_BLOCKS 8

using namespace std;

namespace poa_gpu_utils{

typedef short Edge; 

enum TaskType{ UNDEF = -1, POA_8_32 = 0, POA_8_64 = 1, POA_8_128 = 2, POA_8_255 = 3, 
	       POA_16_32 = 4, POA_16_64 = 5, POA_16_128 = 6, POA_16_255 = 7, 
	       POA_CPU = 8 };

static map<TaskType, int> task_batch_size = {
	{TaskType::UNDEF, -1 },
	{TaskType::POA_8_32, BLOCK_DIM_32 * N_BLOCKS },
	{TaskType::POA_8_64, BLOCK_DIM_64 * N_BLOCKS },
	{TaskType::POA_8_128, BLOCK_DIM_128 * N_BLOCKS },
	{TaskType::POA_8_255, BLOCK_DIM_255 * N_BLOCKS },
	{TaskType::POA_16_32, BLOCK_DIM_16_32 * N_BLOCKS },
	{TaskType::POA_16_64, BLOCK_DIM_16_64 * N_BLOCKS },
	{TaskType::POA_16_128, BLOCK_DIM_16_128 * N_BLOCKS },
	{TaskType::POA_16_255, BLOCK_DIM_16_255 * N_BLOCKS }


};

template<typename T>
class Task{

	public:
		unsigned long long task_id;
		unsigned long task_index;
		T task_data;

		Task(unsigned long long my_id, unsigned long my_idx, T& my_data) : 
			task_id(my_id), task_index(my_idx), task_data(my_data) {}	
	
		Task(unsigned long long my_id, unsigned long my_idx, T&& my_data) : 
			task_id(my_id), task_index(my_idx) { swap(task_data, my_data); }	
					
		Task& operator=(const Task& other){
			task_data = other.task_data;
			task_id = other.task_id;
			task_index = other.task_index;
			return *this;
		}
};

struct TaskRefs{
	
	int uses_global = USES_GLOBAL;
	int max_gapl = MAX_GAPL;
	
	vector<int> nseq_offsets;// = vector<int>(BDIM);
	int tot_nseq = 0;
	char* sequences;
	vector<int> seq_offsets;

	char* result;//[WL * MAXL * BDIM];
	int* res_size;//[BDIM];
	
	int* nseq_offsets_d;
	char* sequences_d;
	char* result_d;
	int* seq_offsets_d;
	int* res_size_d;
	
	int* lpo_edge_offsets_d;
	unsigned char* lpo_letters_d;
	Edge* lpo_edges_d;	
	int* edge_bounds_d;
	unsigned char* end_nodes_d;
	unsigned char* sequence_ids_d;
	
	unsigned char* new_letters_global_d;	
	Edge* new_edges_global_d;
	int* new_edge_bounds_global_d;
	unsigned char* new_end_nodes_global_d;
	unsigned char* new_sequence_ids_global_d;

	unsigned char* dyn_letters_global_d;	
	Edge* dyn_edges_global_d;
	int* dyn_edge_bounds_global_d;
	unsigned char* dyn_end_nodes_global_d;
	unsigned char* dyn_sequence_ids_global_d;

	unsigned char* moves_global_d;	
	short* diagonals_global_sc_d;
	short* diagonals_global_gx_d;
	short* diagonals_global_gy_d;
	int* d_offsets_global_d;
	int* x_to_ys_d;
	int* y_to_xs_d;
	
	int* old_len_global_d;
	int* dyn_len_global_d;	
};

} //end poa_gpu_utils

#endif //POA_GPU_TASKS_H	
