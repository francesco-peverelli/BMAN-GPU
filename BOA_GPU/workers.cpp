#include "workers.h"

namespace poa_gpu_utils{

template void gpu_POA<SEQ_LEN_32, MAX_SEQ_32, WLEN_32, BLOCK_DIM_32>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								     vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);
template void gpu_POA<SEQ_LEN_64, MAX_SEQ_64, WLEN_64, BLOCK_DIM_64>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								     vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  
template void gpu_POA<SEQ_LEN_128, MAX_SEQ_128, WLEN_128, BLOCK_DIM_128>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								         vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  
template void gpu_POA<SEQ_LEN_255, MAX_SEQ_255, WLEN_255, BLOCK_DIM_255>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								         vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  
template void gpu_POA<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32, BLOCK_DIM_16_32>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								     vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);
template void gpu_POA<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64, BLOCK_DIM_16_64>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								     vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  
template void gpu_POA<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128, BLOCK_DIM_16_128>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								         vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  
template void gpu_POA<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255, BLOCK_DIM_16_255>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								         vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  

template void gpu_POA<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32, BLOCK_DIM_32_32>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								     vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);
template void gpu_POA<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64, BLOCK_DIM_32_64>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								     vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  
template void gpu_POA<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128, BLOCK_DIM_32_128>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								         vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  
template void gpu_POA<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255, BLOCK_DIM_32_255>(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, 
								         vector<Task<vector<string>>> &total_res_GPU, int res_write_idx);  


void sel_gpu_POA_alloc(TaskRefs &TR, TaskType& TTy){

	switch(TTy){
		
		case TaskType::POA_8_32:
			gpu_POA_alloc<SEQ_LEN_32, MAX_SEQ_32, WLEN_32, BLOCK_DIM_32>(TR);
			break;
		case TaskType::POA_8_64:
			gpu_POA_alloc<SEQ_LEN_64, MAX_SEQ_64, WLEN_64, BLOCK_DIM_64>(TR);
			break;
		case TaskType::POA_8_128:
                        gpu_POA_alloc<SEQ_LEN_128, MAX_SEQ_128, WLEN_128, BLOCK_DIM_128>(TR);
			break;
		case TaskType::POA_8_255:
                        gpu_POA_alloc<SEQ_LEN_255, MAX_SEQ_255, WLEN_255, BLOCK_DIM_255>(TR);
			break;
		case TaskType::POA_16_32:
			gpu_POA_alloc<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32, BLOCK_DIM_16_32>(TR);
			break;
		case TaskType::POA_16_64:
                        gpu_POA_alloc<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64, BLOCK_DIM_16_64>(TR);
			break;
		case TaskType::POA_16_128:
                        gpu_POA_alloc<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128, BLOCK_DIM_16_128>(TR);
			break;
		case TaskType::POA_16_255:
			gpu_POA_alloc<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255, BLOCK_DIM_16_255>(TR);
			break;
		case TaskType::POA_32_32:
			gpu_POA_alloc<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32, BLOCK_DIM_32_32>(TR);
			break;
		case TaskType::POA_32_64:
                        gpu_POA_alloc<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64, BLOCK_DIM_32_64>(TR);
			break;
		case TaskType::POA_32_128:
                        gpu_POA_alloc<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128, BLOCK_DIM_32_128>(TR);
			break;
		case TaskType::POA_32_255:
			gpu_POA_alloc<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255, BLOCK_DIM_32_255>(TR);
			break;
		default:
			cerr << "Unsupported task type!\n";
			break;
	}
}


void sel_gpu_POA(vector<Task<vector<string>>> &input_tasks, TaskRefs &TR, vector<Task<vector<string>>> &total_res_GPU, 
		 int res_write_idx, TaskType& TTy){

	switch(TTy){
		
		case TaskType::POA_8_32:
			gpu_POA<SEQ_LEN_32, MAX_SEQ_32, WLEN_32, BLOCK_DIM_32>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_8_64:
			gpu_POA<SEQ_LEN_64, MAX_SEQ_64, WLEN_64, BLOCK_DIM_64>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_8_128:
                        gpu_POA<SEQ_LEN_128, MAX_SEQ_128, WLEN_128, BLOCK_DIM_128>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_8_255:
                        gpu_POA<SEQ_LEN_255, MAX_SEQ_255, WLEN_255, BLOCK_DIM_255>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_16_32:
			gpu_POA<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32, BLOCK_DIM_16_32>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_16_64:
                        gpu_POA<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64, BLOCK_DIM_16_64>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_16_128:
                        gpu_POA<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128, BLOCK_DIM_16_128>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_16_255:
                        gpu_POA<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255, BLOCK_DIM_16_255>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_32_32:
			gpu_POA<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32, BLOCK_DIM_32_32>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_32_64:
                        gpu_POA<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64, BLOCK_DIM_32_64>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_32_128:
                        gpu_POA<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128, BLOCK_DIM_32_128>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		case TaskType::POA_32_255:
                        gpu_POA<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255, BLOCK_DIM_32_255>(input_tasks, TR, total_res_GPU, res_write_idx);
			break;
		default:
			cerr << "Unsupported task type!\n";
			break;
	}
}
 
void execute_poa(SyncMultitaskQueues<vector<string>> &t_queues, vector<TaskRefs> &task_refs, 
				 mutex& q_full_mutex, mutex& out_rdy_mutex, 
		                 condition_variable &q_full_var, condition_variable &out_rdy_var, bool &is_notified, bool &flush_mode, 
				 bool &processing_required, TaskType &current_task, TaskType &prev_task, vector<Task<vector<string>>> &result,
				 int num_task_types, size_t &res_write_idx){
	

	unique_lock<mutex> lock(q_full_mutex);

	while(processing_required){

		//cout << "[EXEC_THREAD]: waiting...\n";
		
		q_full_var.wait(lock);
		is_notified = true;

		if(!processing_required) continue;


		if(!flush_mode){
			
			cerr << "Normal execution" << endl;

			if(current_task == TaskType::POA_CPU){
				//call CPU thread worker(s)
				exit(-1);
			} 		

			if(current_task != prev_task){
				//do initialization
				if(prev_task != TaskType::UNDEF){
					gpu_POA_free(task_refs[(int)prev_task], prev_task);
				}
				sel_gpu_POA_alloc(task_refs[(int)current_task], current_task);
			}	

			//do execution
			vector<Task<vector<string>>> input_tasks;
			cerr << "Retrieving data.." << endl;
			t_queues.retrieve_data_batch(input_tasks, current_task);
			
			cerr << "Exec poa" << endl;

			sel_gpu_POA(input_tasks, task_refs[(int)current_task], result, res_write_idx, current_task);
			
			cerr << "Exec done " << endl;

			prev_task = current_task;
			res_write_idx += input_tasks.size();
			//cout << "T time = " << duration_T.count() << " microseconds" << endl;
			//cout << "A time = " << duration_A.count() << " microseconds" << endl;
			
		}else{

			//cerr << "Flush" << endl;

			if(prev_task != TaskType::UNDEF){
				gpu_POA_free(task_refs[prev_task], prev_task);
			}
			
			bool prev_batch_executed = false;
			for(int TTy = 0; TTy < num_task_types; TTy++){
				TaskType t_type = (TaskType)TTy;
				if(prev_batch_executed){
					gpu_POA_free(task_refs[TTy-1], (TaskType)(TTy-1));
					prev_batch_executed = false;
				}
				if(t_queues.get_queue_size(t_type) != 0){
					prev_batch_executed = true;
					sel_gpu_POA_alloc(task_refs[TTy], t_type);
					vector<Task<vector<string>>> input_tasks;
					t_queues.retrieve_data_batch(input_tasks, t_type);


					sel_gpu_POA(input_tasks, task_refs[TTy], result, res_write_idx, t_type);
					res_write_idx += input_tasks.size();
				
					//cout << "T time = " << duration_T.count() << " microseconds" << endl;
					
				}
			}
			cudaDeviceReset();
			
			t_queues.reset_combined_size();
			res_write_idx = 0;
			out_rdy_var.notify_all();

			
		}//flush queue code end
		
		is_notified = false;
	} //processing_required loop end


}


} //end poa_gpu_utils

