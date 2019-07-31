#ifndef BMEAN
#define BMEAN



#include <vector>
#include <string>
#include "utils.h"
#include "workers.h"
#include <cmath>


extern "C"{
#include "lpo.h"
#include "msa_format.h"
#include "align_score.h"
#include "default.h"
#include "poa.h"
#include "seq_util.h"
}




using namespace std;

pair<vector<vector<string>>, unordered_map<kmer, unsigned>> MSABMAAC(const vector<string>& nadine,uint32_t la,double cuisine, unsigned solidThresh, unsigned minAnchors, unsigned maxMSA, string path);

void gpu_global_consensus_worker(int id, vector<vector<string>>& res, vector<vector<string>>& V, unsigned maxMSA, string path);

void poa_executor_worker(int id, poa_gpu_utils::SyncMultitaskQueues<vector<string>> &t_queues, vector<poa_gpu_utils::TaskRefs> &task_refs,
		                 mutex& q_full_mutex, mutex& out_rdy_mutex, 
		                 condition_variable &q_full_var, condition_variable &out_rdy_var, bool &is_notified, bool &flush_mode, 
				 bool &processing_required, poa_gpu_utils::TaskType &current_task, poa_gpu_utils::TaskType &prev_task,
				 vector<poa_gpu_utils::Task<vector<string>>> &result,
				 int num_task_types, std::size_t &res_write_idx);

vector<string> consensus_POA( vector<string>& W, unsigned maxMSA, string path);

vector<vector<string>> global_consensus(const  vector<vector<string>>& V, uint32_t n, unsigned maxMSA, string path);

vector<vector<string>> global_consensus_gpu(const  vector<vector<string>>& V, uint32_t n, unsigned maxMSA, string path);

#endif
