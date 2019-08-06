#ifndef TASKS_MANAGER_H
#define TASKS_MANAGER_H

#include "poa-gpu-tasks.h"
#include<condition_variable>
#include<iostream>

#define TIMEOUT 10

using namespace std;

namespace poa_gpu_utils{

template<typename T> 
inline TaskType get_task_type(Task<T> &task);

template<typename T>
class SyncMultitaskQueues{

	private:
		vector<vector<Task<T>>> gpu_task_queues;	
		mutable mutex q_mutex;
		condition_variable &queue_full_var;
		bool &exec_notified;
		bool &flush_mode;
		TaskType &curr_task;
		int n_task_types;
		size_t combined_size;

	public:
		SyncMultitaskQueues(std::condition_variable &qf, TaskType& ct, int n_t_types, bool &exec_ntfy, bool &f_mode) : 
				    queue_full_var(qf), curr_task(ct), exec_notified(exec_ntfy), flush_mode(f_mode) {
			n_task_types = n_t_types;
			gpu_task_queues = vector<vector<Task<T>>>(n_t_types);
			combined_size = 0;
		}		

		void enqueue_task_vector(vector<Task<T>> &tasks, vector<TaskType> &task_types){
				
				lock_guard<mutex> lock(q_mutex);

				int n = tasks.size();
				bool notify = false;
				
				int Ti = 0;
				for(Task<T> &t : tasks){
					t.task_id = combined_size + Ti;
					gpu_task_queues[(int)task_types[Ti]].push_back(t);
					Ti++;	
				}
				combined_size += n;

				if(!exec_notified){
					for(auto Tty : task_types){
						if(gpu_task_queues[(int)Tty].size() > task_batch_size[Tty]){
							flush_mode = false;
							curr_task = Tty;
							notify  = true;
							break;
						}
					}
				}

				if(notify){
					while(!exec_notified)
						queue_full_var.notify_one(); 	
				}

				return;
		}

		void wait_and_flush_queue(){

			cout << "Flush called\n";
			mutex my_mutex;
			unique_lock<mutex> my_lock(my_mutex);
			condition_variable my_var;

			while(exec_notified){
				cout << "Waiting for exec to terminate...\n";
				my_var.wait_for(my_lock, chrono::duration<int>(TIMEOUT), [&]{ return exec_notified == 0; });
			}

			lock_guard<mutex> lock(q_mutex);

			flush_mode = true;
			while(!exec_notified){
				cout << "Exec notified\n";
				queue_full_var.notify_one();
			}
		}
	
		void retrieve_data_batch(vector<Task<T>> &tasks, TaskType &task_type){
			lock_guard<mutex> lock(q_mutex);
			tasks.swap(gpu_task_queues[task_type]);
			gpu_task_queues[task_type].clear();
		}
	
		size_t get_combined_size(){ 
			lock_guard<mutex> lock(q_mutex); 
			return combined_size; 
		}
		
		int get_queue_size(TaskType &t){ return gpu_task_queues[t].size(); }
		
		void reset_combined_size(){
			lock_guard<mutex> lock(q_mutex); 
			combined_size = 0; 
		}
};

template<typename T>
class SyncMultitaskConcurrencyManager{

private:

	//poa task queues
	poa_gpu_utils::SyncMultitaskQueues<T> *poa_queues;

public:

	//vector to hold results
	vector<poa_gpu_utils::Task<T>> results;
	//condition variable for the queues
	condition_variable queue_rdy_var;
	mutex queue_rdy_mutex;
	
	condition_variable output_rdy_var;
	mutex output_rdy_mutex;

	mutex task_reg_mutex;

	//current and previous task type executed 
	poa_gpu_utils::TaskType current_task = poa_gpu_utils::TaskType::UNDEF;
	poa_gpu_utils::TaskType previous_task = poa_gpu_utils::TaskType::UNDEF;
	vector<TaskRefs> task_refs;

	std::size_t current_res_index = 0;
	int num_task_types;
	bool processing_required = true;
	bool exec_notified = false;
	bool flush_mode = false;
	
	SyncMultitaskConcurrencyManager(int n_task_types, int batch_size) : num_task_types(n_task_types) {
		task_refs = vector<TaskRefs>(n_task_types);
		poa_queues = new poa_gpu_utils::SyncMultitaskQueues<T>(
					queue_rdy_var, current_task, n_task_types, exec_notified, flush_mode
     	             	    	 );
	}
	
	~SyncMultitaskConcurrencyManager(){ delete poa_queues; }	

	void enqueue_task_vector(vector<Task<T>> &tasks, vector<TaskType> &task_types){
		size_t new_size = poa_queues->get_combined_size() + tasks.size();
		if(new_size > results.size()){
			results.resize(new_size, poa_gpu_utils::Task<vector<string>>(0,0,vector<string>()));
		}
		poa_queues->enqueue_task_vector(tasks, task_types);
	}

	void wait_and_flush_queue(){
		poa_queues->wait_and_flush_queue();
	}

	poa_gpu_utils::SyncMultitaskQueues<T>& get_queues_ref(){
		return *poa_queues;
	}

	/*void wait_and_flush_all(){
		condition_variable all_enqueued;
		unique_lock<mutex> lck(output_rdy_mutex);
		while(preprocessing_tasks != 0){
			cout << "waiting at all enqueued\n";
			all_enqueued.wait_for(lck, chrono::duration<int>(TIMEOUT), [&]{ return preprocessing_tasks == 0; });
		}
		queue_rdy_var.notify_one();
		output_rdy_var.wait(lck);
		processing_required = false;
	}*/
};

template<>
inline TaskType get_task_type<vector<string>>(Task<vector<string>> &task){
	
	vector<string> &task_data = task.task_data;
	int W = task_data.size();
	int max_s = 0;
	if(W <= 8){
		for(auto s : task_data){
			int sz = s.size();
			if(max_s < sz){
				max_s = sz;
			}
		}
		if(max_s <= SEQ_LEN_32)
			return TaskType::POA_8_32;
		else if(max_s <= SEQ_LEN_64)
			return TaskType::POA_8_64;
		else if(max_s <= SEQ_LEN_128)
			return TaskType::POA_8_128;
		else if(max_s <= SEQ_LEN_255)
			return TaskType::POA_8_255;
		else
			return TaskType::POA_CPU;

	}else if(W <= 16){
		for(auto s : task_data){
			int sz = s.size();
			if(max_s < sz){
				max_s = sz;
			}
		}
		if(max_s <= SEQ_LEN_16_32)
			return TaskType::POA_16_32;
		else if(max_s <= SEQ_LEN_16_64)
			return TaskType::POA_16_64;
		else if(max_s <= SEQ_LEN_16_128)
			return TaskType::POA_16_128;
		else if(max_s <= SEQ_LEN_16_255)
			return TaskType::POA_16_255;
		else
			return TaskType::POA_CPU;

	}else{
		return TaskType::POA_CPU;
	}
}

} //end poa_gpu_utils

#endif //TASKS_MANAGER_H
