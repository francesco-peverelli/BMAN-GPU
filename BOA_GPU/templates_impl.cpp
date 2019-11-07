//  intializations header file

/*** Explicit instantiations ***/ 

template __global__ void assign_device_memory<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_32, MAX_SEQ_32, WLEN_32>(int* d_ptr, const int num_blocks);
	
//----------------------------------------------------------------------------------------------//


template __global__ void assign_device_memory<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_64, MAX_SEQ_64, WLEN_64>(int* d_ptr, const int num_blocks);
	
//----------------------------------------------------------------------------------------------//

template __global__ void assign_device_memory<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_128, MAX_SEQ_128, WLEN_128>(int* d_ptr, const int num_blocks);
	
	
//----------------------------------------------------------------------------------------------//


template __global__ void assign_device_memory<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_255, MAX_SEQ_255, WLEN_255>(int* d_ptr, const int num_blocks);
	

template __global__ void assign_device_memory<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_16_32, MAX_SEQ_16_32, WLEN_16_32>(int* d_ptr, const int num_blocks);
	
//----------------------------------------------------------------------------------------------//


template __global__ void assign_device_memory<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_16_64, MAX_SEQ_16_64, WLEN_16_64>(int* d_ptr, const int num_blocks);
	
//----------------------------------------------------------------------------------------------//

template __global__ void assign_device_memory<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_16_128, MAX_SEQ_16_128, WLEN_16_128>(int* d_ptr, const int num_blocks);
	
	
//----------------------------------------------------------------------------------------------//


template __global__ void assign_device_memory<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_16_255, MAX_SEQ_16_255, WLEN_16_255>(int* d_ptr, const int num_blocks);
	

template __global__ void assign_device_memory<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_32_32, MAX_SEQ_32_32, WLEN_32_32>(int* d_ptr, const int num_blocks);
	
//----------------------------------------------------------------------------------------------//


template __global__ void assign_device_memory<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_32_64, MAX_SEQ_32_64, WLEN_32_64>(int* d_ptr, const int num_blocks);
	
//----------------------------------------------------------------------------------------------//

template __global__ void assign_device_memory<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_32_128, MAX_SEQ_32_128, WLEN_32_128>(int* d_ptr, const int num_blocks);
	
	
//----------------------------------------------------------------------------------------------//


template __global__ void assign_device_memory<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

template  __global__ void init_diagonals<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

template  __global__ void sw_align<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
template  __device__ void trace_back_lpo_alignment<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

template  __global__ void compute_d_offsets<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_new_lpo_size<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

template  __global__ void fuse_lpo<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

template  __global__ void copy_new_lpo_data<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int j_seq_idx, int* nseq_offsets);

template  __global__ void compute_edge_offsets<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int* seq_offsets, int* nseq_offsets);

template  __global__ void generate_lpo<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

template  __global__ void copy_result_sizes<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int *nseq_offsets, int* res_size);

template  __global__ void compute_result<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

template  __global__ void suffix_sum<SEQ_LEN_32_255, MAX_SEQ_32_255, WLEN_32_255>(int* d_ptr, const int num_blocks);
	

