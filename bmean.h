#ifndef BMEAN
#define BMEAN



#include <vector>
#include <string>
#include "utils.h"
#include "workers.h"
#include <cmath>
#include "../../CTPL/ctpl_stl.h"

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

pair<vector<poa_gpu_utils::Task<vector<string>>>, unordered_map<kmer, unsigned>> 
MSABMAAC_gpu_enqueue(int id, const vector<string>& nadine,uint32_t la,double cuisine, unsigned solidThresh, unsigned minAnchors, unsigned maxMSA, string path);

pair<vector<poa_gpu_utils::Task<vector<string>>>, unordered_map<kmer, unsigned>>
MSABMAAC_gpu_enqueue_ctpl(int id, const vector<string>& nadine,uint32_t la,double cuisine, unsigned solidThresh, unsigned minAnchors, unsigned maxMSA, string path);

void MSABMAAC_gpu_flush();

void MSABMAAC_gpu_init_ctpl(size_t batch_size, ctpl::thread_pool &my_pool);

void MSABMAAC_gpu_init_batch(size_t batch_size, thread &exec_t);

void MSABMAAC_gpu_init(size_t batch_size);

void MSABMAAC_gpu_done();

vector<vector<string>> MSABMAAC_gpu_dequeue(vector<poa_gpu_utils::Task<vector<string>>> &task_vector, unsigned maxMSA, string path); 

vector<vector<string>> MSABMAAC_gpu_dequeue_ctpl(int id, vector<poa_gpu_utils::Task<vector<string>>> &task_vector, unsigned maxMSA, string path);      
bool needs_poa(const vector<string>& V);

#endif
