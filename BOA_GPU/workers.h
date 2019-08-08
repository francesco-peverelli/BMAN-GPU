#ifndef WORKERS_H
#define WORKERS_H

#include "poa-gpu.h"
#include <chrono>

#define NOW high_resolution_clock::now() 

namespace poa_gpu_utils{

//void sel_gpu_POA_alloc(TaskRefs &TR, TaskType& TTy);

//void sel_gpu_POA(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, vector<Task<vector<string>>> &total_res_GPU, 
//		 int res_write_idx, TaskType& TTy);

void execute_poa(SyncMultitaskQueues<vector<string>> &t_queues, vector<TaskRefs> &task_refs, 
				 mutex& q_full_mutex, mutex& out_rdy_mutex, 
		                 condition_variable &q_full_var, condition_variable &out_rdy_var, bool &is_notified, bool &flush_mode, 
				 bool &processing_required, TaskType &current_task, TaskType &prev_task, vector<Task<vector<string>>> &result,
				 int num_task_types, size_t &res_write_idx);
} //end poa_gpu_utils

#endif  //WORKERS_H 
