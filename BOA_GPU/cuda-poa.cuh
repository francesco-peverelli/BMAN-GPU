#ifndef CUDA_POA_H
#define CUDA_POA_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "poa-constants.h"


static const int NOT_ALIGNED = -1;

typedef short Edge; 

/*struct Edge {
	//int score;
	short ipos;
};*/

/*struct Move {
	unsigned char x;
	unsigned char y;
};*/

/*struct ScoreCell {
	short score;
	//positions of gap penalties in gaps array
	short gap_x;
	short gap_y;
};*/

struct MaxCell {
	int val;
	int x;
	int y;
};

//scoring scheme
__device__ inline int score(int i, int j);
__device__ inline int gap_penalty_x(int pos, int max_gapl, int uses_global);
__device__ inline int gap_penalty_y(int pos, int max_gapl, int uses_global);
__device__ inline int next_gap(int i, int max_gapl, int uses_global);
__device__ inline int next_perp_gap(int i, int max_gapl, int uses_global);

template<int N_WRAPS>
__inline__ __device__ MaxCell blockReduceMax(MaxCell cell);

template<int SL, int MAXL, int WL>
__global__ void assign_device_memory(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template<int SL, int MAXL, int WL>
__global__ void suffix_sum(int* d_ptr, const int num_blocks);

template<int SL, int MAXL, int WL>
__global__ void init_diagonals(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

//alignment kernel
template<int SL, int MAXL, int WL>
__global__ void sw_align(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 

//traceback
template<int SL, int MAXL, int WL> 
__device__ void trace_back_lpo_alignment(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

// diagonal offsets computation kernel
template<int SL, int MAXL, int WL>
__global__ void compute_d_offsets(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

//new lpo size kernel 
template<int SL, int MAXL, int WL>
__global__ void compute_new_lpo_size(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

//lpo fusion kernel
template<int SL, int MAXL, int WL>
__global__ void fuse_lpo(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template<int SL, int MAXL, int WL>
extern __global__ void copy_new_lpo_data(int j_seq_idx, int* nseq_offsets);

template<int SL, int MAXL, int WL>
extern __global__ void compute_edge_offsets(int* seq_offsets, int* nseq_offsets);

//lpo generation kernel
template<int SL, int MAXL, int WL>
__global__ void generate_lpo(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template<int SL, int MAXL, int WL>
__global__ void copy_result_sizes(int *nseq_offsets, int* res_size);

//final alignment result computation
template<int SL, int MAXL, int WL>
__global__ void compute_result(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template<int SL, int MAXL, int WL>
__device__ void debug_print_lpo(int len, unsigned char* seq, Edge* edge, int* start, unsigned char* endlist, unsigned char* seq_ids, int nseq);

#endif /*CUDA-POA*/
