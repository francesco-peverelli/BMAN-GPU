#include "cuda-poa.cuh"

__device__ int score(unsigned char i, unsigned char j) { return i == j ? MATCH : MISMATCH; }
__device__ int gap_penalty_x(int pos, int max_gapl, int uses_global) { return (uses_global == 0 && pos == max_gapl + 1) ? 0 : GAP; }
__device__ int gap_penalty_y(int pos, int max_gapl, int uses_global) { return (uses_global == 0 && pos == max_gapl + 1) ? 0 : GAP; }
__device__ int next_gap(int i, int max_gapl, int uses_global) { return (i < max_gapl) ? (i + 1) : i == max_gapl ? i : uses_global ? 1 : max_gapl + 1; }
__device__ int next_perp_gap(int i, int max_gapl, int uses_global) { return  (i < max_gapl) ? (i + 1) : i == max_gapl ? i : uses_global ? 1 : max_gapl + 1; }

__device__ int* lpo_offsets;
__device__ int* lpo_edge_offsets;
__device__ unsigned char* lpo_letters;
__device__ Edge* lpo_edges;
__device__ int* edge_bounds;
__device__ unsigned char* end_nodes;
__device__ unsigned char* sequence_ids;

__device__ int* old_len_global;
__device__ unsigned char* new_letters_global;
__device__ Edge* new_edges_global;
__device__ int* new_edge_bounds_global;
__device__ unsigned char* new_end_nodes_global;
__device__ unsigned char* new_sequence_ids_global;

__device__ int* dyn_len_global;
__device__ unsigned char* dyn_letters_global;
__device__ Edge* dyn_edges_global;
__device__ int* dyn_edge_bounds_global;
__device__ unsigned char* dyn_end_nodes_global;
__device__ unsigned char* dyn_sequence_ids_global;

__device__ unsigned char* moves_x_global;
__device__ unsigned char* moves_y_global;
__device__ short* diagonals_sc_global;
__device__ short* diagonals_gx_global;
__device__ short* diagonals_gy_global;
__device__ int* d_offsets_global;
__device__ int* x_to_ys;
__device__ int* y_to_xs;
__device__ int g_space_exceeded = 0;

__inline__ __device__ MaxCell wrapReduceMax(MaxCell cell){
	
	for(int offset = wrapSize / 2; offset > 0; offset /= 2){
		short val = __shfl_down_sync(FULL_MASK, cell.val, offset);
		int x = __shfl_down_sync(FULL_MASK, cell.x, offset);
		int y = __shfl_down_sync(FULL_MASK, cell.y, offset);
		int is_max = val >= cell.val && ( val > cell.val | x < cell.x | y < cell.y );
		cell.val = is_max ? val : cell.val;
		cell.x = is_max ? x : cell.x;
		cell.y = is_max ? y : cell.y;
	}
	return cell;
}

template<int N_WRAPS>
__inline__ __device__ MaxCell blockReduceMax(MaxCell cell){
	
	static __shared__ MaxCell shared_max[N_WRAPS];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	
	cell = wrapReduceMax(cell);
	
	if(lane == 0){
		shared_max[wid] = cell;	
	}
	__syncthreads();

	MaxCell zero_cell = { -999999, -1, -1 };
	cell = (threadIdx.x < blockDim.x / warpSize) ? shared_max[lane] : zero_cell;
	
	if(wid == 0) {
		cell = wrapReduceMax(cell);
	}
	return cell;
}

template<int> __inline__ __device__ MaxCell blockReduceMax(MaxCell cell);

template<int SL, int MAXL, int WL>
 __global__ void assign_device_memory(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks){
	
	lpo_edge_offsets = ledges_offs;
	lpo_letters = lletters;
	lpo_edges = ledges;
	edge_bounds = ebounds;
	end_nodes = ennodes;
	sequence_ids = seq_ids;

	new_letters_global = nletters;
	new_edges_global = nedges;
	new_edge_bounds_global = nedgebounds;
	new_end_nodes_global = n_end_nodes;
	new_sequence_ids_global = n_seq_ids;
	
	dyn_letters_global = dletters;
	dyn_edges_global = dedges;
	dyn_edge_bounds_global = dedgebounds;
	dyn_end_nodes_global = d_end_nodes;
	dyn_sequence_ids_global = d_seq_ids;
	
	moves_x_global = moves;
	moves_y_global = moves + (unsigned long)(MAXL+1)*(SL+1) * num_blocks;
	diagonals_sc_global = diagonals_sc;
	diagonals_gx_global = diagonals_gx;
	diagonals_gy_global = diagonals_gy;
	d_offsets_global = d_offs;
	x_to_ys = xy;
	y_to_xs = yx;

	old_len_global = oldlg;
	dyn_len_global = dynlg;
}

template<int SL, int MAXL, int WL>
 __global__ void suffix_sum(int* d_ptr, const int num_blocks){
	int offset = 0;
	for(int i = 0; i < num_blocks; i++){
		offset += d_ptr[i]; 
		d_ptr[i] = offset;
	}
}

template<int SL, int MAXL, int WL>
 __device__ void trace_back_lpo_alignment(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left,
					 int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets) {

	int xmove, ymove;
	Edge left;
	
	for (int i = 0; i < len_x; i++) {
		x_to_y[i] = (int)NOT_ALIGNED;
	}

	for (int i = 0; i < len_y; i++) {
		y_to_x[i] = (int)NOT_ALIGNED;
	}

	while (best_x >= 0 && best_y >= 0) {
		
		int diagonal = best_x + best_y + 2;
		int offset = (diagonal <= len_y) ? best_x+1 : best_x+1 - (diagonal - len_y); 
		xmove = (move_x+d_offsets[diagonal])[offset];
		ymove = (move_y+d_offsets[diagonal])[offset];

		if (xmove > 0 && ymove > 0) { /* ALIGNED! MAP best_x <--> best_y */
			x_to_y[best_x] = best_y;
			y_to_x[best_y] = best_x;
		}

		if (xmove == 0 && ymove == 0) { /* FIRST ALIGNED PAIR */
			x_to_y[best_x] = best_y;
			y_to_x[best_y] = best_x;
			break;  /* FOUND START OF ALIGNED REGION, SO WE'RE DONE */
		}

		if (xmove > 0) { /* TRACE BACK ON X */
			int start = start_x[best_x];
			while ((--xmove) > 0) {
				start++;
			}
			left = x_left[start];
			best_x = left;
		}

		if (ymove > 0) { /* TRACE BACK ON Y */
			int start = start_y[best_y];
			while ((--ymove) > 0) {
				start++;
			}
			left = y_left[start];
			best_y = left;
		}
	}
	return;
}

template<int SL, int MAXL, int WL>
 __global__ void init_diagonals(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets){
	
	if(g_space_exceeded) return;

	int nseq;
	int block_offset;
	int myId = blockIdx.x;
	int y_seq_offs;
	int y_edge_offs;
	int len_y;
	int len_x;
	
	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = dyn_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}

	if(j_seq_idx < nseq){
	
		y_seq_offs = lpo_offsets[block_offset + j_seq_idx-1];
		y_edge_offs = lpo_edge_offsets[block_offset + j_seq_idx-1];
		len_y = lpo_offsets[block_offset + j_seq_idx] - lpo_offsets[block_offset + j_seq_idx-1];
	
		int global_seq_idx_y = block_offset + j_seq_idx;
		Edge* left_x = dyn_edges_global + MAXL * EDGE_F * myId; 
		Edge* left_y = lpo_edges + y_edge_offs;
		int* lx_start = dyn_edge_bounds_global + (MAXL+1) * myId;
		int* ly_start = edge_bounds + y_seq_offs + global_seq_idx_y;

		short* diagonals_sc = diagonals_sc_global + (MAXL+1)*(SL+1) * myId;
		short* diagonals_gx = diagonals_gx_global + (MAXL+1)*(SL+1) * myId;
		short* diagonals_gy = diagonals_gy_global + (MAXL+1)*(SL+1) * myId;
		int* d_offsets = d_offsets_global + (MAXL+SL+1) * myId;
	
		int min_d = len_x < len_y ? len_x : len_y;

		diagonals_sc[0] = 0;
		diagonals_gx[0] = max_gapl + 1;
		diagonals_gy[0] = max_gapl + 1;

		int min_score = -999999;
		int try_score;   

		for (int i = 1; i < len_x + 1; i++) {

			int offs = i < min_d ? i : min_d;
			short &curr_cell_sc = (diagonals_sc+d_offsets[i])[offs];
			short &curr_cell_gx = (diagonals_gx+d_offsets[i])[offs];
			short &curr_cell_gy = (diagonals_gy+d_offsets[i])[offs];
			curr_cell_sc = min_score;

			int k = lx_start[i - 1];
			for (int x_count = 1; k < lx_start[i]; k++, x_count++) {

				Edge xl = left_x[k];
				int prev_last_cell = xl + 1 < min_d ? xl + 1 : min_d;
				short prev_sc = (diagonals_sc + d_offsets[xl + 1])[prev_last_cell];
				short prev_gx = (diagonals_gx + d_offsets[xl + 1])[prev_last_cell];
		
				try_score = prev_sc - gap_penalty_x(prev_gx, max_gapl, uses_global);    

				if (try_score > curr_cell_sc) {
					curr_cell_sc = try_score;
					curr_cell_gx = next_gap(prev_gx, max_gapl, uses_global);
					curr_cell_gy = next_perp_gap(prev_gx, max_gapl, uses_global);
				}
			}
		}

		for (int i = 1; i < len_y + 1; i++) {

			short &curr_cell_sc = diagonals_sc[d_offsets[i]];
			short &curr_cell_gx = diagonals_gx[d_offsets[i]];
			short &curr_cell_gy = diagonals_gy[d_offsets[i]];
			curr_cell_sc = min_score;

			int k = ly_start[i - 1];
			for (int y_count = 1; k < ly_start[i]; k++, y_count++) {

				Edge yl = left_y[k];
				short prev_sc = diagonals_sc[d_offsets[yl + 1]];
				short prev_gy = diagonals_gy[d_offsets[yl + 1]];

				try_score = prev_sc - gap_penalty_y(prev_gy, max_gapl, uses_global);      

				if (try_score > curr_cell_sc) {
					curr_cell_sc = try_score;
					curr_cell_gx = next_perp_gap(prev_gy, max_gapl, uses_global);
					curr_cell_gy = next_gap(prev_gy, max_gapl, uses_global);
				}
			}
		}
	}
}

template<int SL, int MAXL, int WL>
 __global__ void sw_align(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets) {
	
	if(g_space_exceeded) return;

	int myId = blockIdx.x;
	__shared__ int nseq;
	__shared__ int block_offset;
	__shared__ int y_seq_offs;
	__shared__ int y_edge_offs;
	__shared__ int len_y;
	__shared__ int len_x;
	
	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = dyn_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}
	
	if(j_seq_idx < nseq){
	
	y_seq_offs = lpo_offsets[block_offset + j_seq_idx-1];
	y_edge_offs = lpo_edge_offsets[block_offset + j_seq_idx-1];
	len_y = lpo_offsets[block_offset + j_seq_idx] - lpo_offsets[block_offset + j_seq_idx-1];
	
	int global_seq_idx_y = block_offset + j_seq_idx;
	
	int c = threadIdx.x;

	__shared__ unsigned char* seq_x;
	__shared__ unsigned char* end_list_x;
	__shared__ unsigned char* end_list_y;
	__shared__ int* x_to_y;
	__shared__ int* y_to_x;
	__shared__ short* diagonals_sc;
	__shared__ short* diagonals_gx;
	__shared__ short* diagonals_gy;
	__shared__ unsigned char* moves_x;
	__shared__ unsigned char* moves_y;

	if(c==0){
		seq_x = dyn_letters_global + MAXL * myId;
		end_list_x = dyn_end_nodes_global + MAXL * myId;
		end_list_y = end_nodes + y_seq_offs;
		x_to_y= x_to_ys + MAXL * myId;
		y_to_x = y_to_xs + MAXL * myId;
		diagonals_sc = diagonals_sc_global + (MAXL+1)*(SL+1) * myId;
		diagonals_gx = diagonals_gx_global + (MAXL+1)*(SL+1) * myId; 
		diagonals_gy = diagonals_gy_global + (MAXL+1)*(SL+1) * myId;
		moves_x = moves_x_global + (MAXL+1)*(SL+1) * myId;
		moves_y = moves_y_global + (MAXL+1)*(SL+1) * myId;
	}
	__syncthreads();

	static __shared__ int lx_start[MAXL+1];
	static __shared__ int ly_start[SL+1];
	static __shared__ int d_offsets[MAXL+SL+1];
	static __shared__ Edge left_x[MAXL*EDGE_F];
	static __shared__ Edge left_y[SL*EDGE_F];
	static __shared__ unsigned char seq_y[SL];
	
	int offs = 0;
	do{
		if(offs + c < len_x+1){
			lx_start[offs + c] = (dyn_edge_bounds_global + (MAXL+1) * myId)[offs+c];
		}	
		offs += SL+1;
	}while(offs < len_x+1);

	if(c < len_y+1){
		ly_start[c] = (edge_bounds + y_seq_offs + global_seq_idx_y)[c];
	}
	__syncthreads();
	int x_left_dim = lx_start[len_x];
	int y_left_dim = ly_start[len_y];

	offs = 0;
	do{
		if(offs + c < x_left_dim){
			left_x[offs+c] = (dyn_edges_global + MAXL * EDGE_F * myId)[offs+c];
		}
		offs +=SL+1;
	}while(offs < x_left_dim);

	offs = 0;
	do{
		if(offs + c < y_left_dim){
			left_y[offs+c] = (lpo_edges + y_edge_offs)[offs+c];
		}
		offs +=SL+1;
	}while(offs < y_left_dim);
	
	if(c < len_y){
		seq_y[c] = (lpo_letters + y_seq_offs)[c];
	}

	offs = 0;
	do{
		if(offs + c < len_x+len_y+1){
			d_offsets[c+offs] = (d_offsets_global + (MAXL+SL+1) * myId)[c+offs];
		}
		offs += SL+1;
	}while(offs < len_x+len_y+1);
	
	MaxCell max = { -999999, -1, -1 };
	int min_d = len_x < len_y ? len_x : len_y;

	__syncthreads();
	
	for (int n = 2; n < len_x + len_y + 1; n++) {

		int lower_bound = n <= len_y;
		int upper_bound = min_d+lower_bound < n ? min_d+lower_bound : n;
		upper_bound = upper_bound < len_x + len_y + 1 - n ? upper_bound : len_x + len_y + 1 - n;
		
		if (c >= lower_bound && c < upper_bound) {

			int match_score = ((uses_global == 0)-1) & (-999999);
			int match_x = 0;
			int match_y = 0;

			int insert_x_score = -999999;
			int insert_y_score = -999999;
			int insert_x_x = 0;
			int insert_y_y = 0;
			int insert_x_gap = 0;
			int insert_y_gap = 0;

			int try_score = -999999;
			
			int j = c + (((n - len_y < 0)-1) & (n - len_y));
			int i = n - j;

			int possible_end_cell = uses_global == 0 ||
				(end_list_x[j - 1] == 0 && end_list_y[i - 1] == 0); 

			int k = ly_start[i-1];
			for (int y_count = 1; k < ly_start[i]; k++, y_count++) {

				int i_prev = left_y[k] + 1;
				int k = ((0 > i_prev + j - len_y)-1) & (i_prev + j - len_y);
				int n_prev = i_prev + j;
				int c_prev = j - k;
				int prev_gy;

				try_score = (diagonals_sc + d_offsets[n_prev])[c_prev];
				prev_gy = (diagonals_gy + d_offsets[n_prev])[c_prev];
				try_score -= gap_penalty_y(prev_gy, max_gapl, uses_global);
				
				if (try_score > insert_y_score) {
					insert_y_score = try_score;
					insert_y_y = y_count;
					insert_y_gap = prev_gy;
				}

			}

			k = lx_start[j - 1];
			for (int x_count = 1; k < lx_start[j]; k++, x_count++) {

				Edge xl = left_x[k];
				int j_prev = xl + 1;
				int k = ((0 > j_prev + i - len_y)-1) & (j_prev + i - len_y);
				int n_prev = j_prev + i;
				int c_prev = j_prev - k;
				int prev_gx;

				try_score = (diagonals_sc + d_offsets[n_prev])[c_prev];
				prev_gx = (diagonals_gx + d_offsets[n_prev])[c_prev];
				try_score -= gap_penalty_x(prev_gx, max_gapl, uses_global);
				
				if (try_score > insert_x_score) {
					insert_x_score = try_score;
					insert_x_x = x_count;
					insert_x_gap = prev_gx;
				}

				k = ly_start[i-1];
				for (int y_count = 1; k < ly_start[i]; k++, y_count++) {

					int i_prev = left_y[k] + 1;
					int k = ((0 > i_prev + j_prev - len_y)-1) & (i_prev + j_prev - len_y);
					int n_prev = i_prev + j_prev;
					int c_prev = j_prev - k;

					try_score = (diagonals_sc + d_offsets[n_prev])[c_prev];	

					if (try_score > match_score) {
						match_score = try_score;
						match_x = x_count;
						match_y = y_count;
					}
				}
			}

			match_score += score(seq_x[j - 1], seq_y[i - 1]); 
			
			unsigned char my_move_x; 
			unsigned char my_move_y; 
			
			short my_score;
			short my_gx; 
			short my_gy; 
			
			int match_mask = (match_score <= insert_y_score || match_score <= insert_x_score)-1;
			int ins_x_mask = (insert_x_score < match_score || insert_x_score <= insert_y_score)-1;
			int ins_y_mask = (insert_y_score < match_score || insert_y_score < insert_x_score)-1;

			my_score = (match_score & match_mask) + (insert_x_score & ins_x_mask) + (insert_y_score & ins_y_mask);
			
			my_gx = 0 + (next_gap(insert_x_gap, max_gapl, uses_global) & ins_x_mask) + 
				     (next_perp_gap(insert_y_gap, max_gapl, uses_global) & ins_y_mask);
			
			my_gy = 0 + (next_perp_gap(insert_x_gap, max_gapl, uses_global) & ins_x_mask) + 
				     (next_gap(insert_y_gap, max_gapl, uses_global) & ins_y_mask);
			
			my_move_x = (match_x & match_mask) + (insert_x_x & ins_x_mask) + 0;
			
			my_move_y = (match_y & match_mask) + (insert_y_y & ins_y_mask) + 0;
			
			if (possible_end_cell && my_score >= max.val) {
				if (my_score > max.val ||
					(j-1 == max.x && i-1 < max.y) || j-1 < max.x) {
					max.val = my_score;
					max.x = j-1;
					max.y = i-1;
				}
			}
			(moves_x+d_offsets[n])[c] = my_move_x;
			(moves_y+d_offsets[n])[c] = my_move_y;
			(diagonals_sc + d_offsets[n])[c] = my_score;
			(diagonals_gx + d_offsets[n])[c] = my_gx;
			(diagonals_gy + d_offsets[n])[c] = my_gy;			
				
		}
		__syncthreads();

	}

	max = blockReduceMax<(SL+1) / wrapSize>(max);
	
	if(c==0){
		trace_back_lpo_alignment<SL, MAXL, WL>(len_x, len_y, moves_x, moves_y, left_x, left_y, lx_start, ly_start, 
							     max.x, max.y, x_to_y, y_to_x, d_offsets);
	}

	}//thread execution if-end
	
	return;
}

__device__ void debug_print_lpo(int len, unsigned char* seq, Edge* edge, int* start, unsigned char* endlist, unsigned char* seq_ids, int nseq){

	printf("lpo-print: len=%d, seq: ", len);
	for(int i = 0; i < len; i++){
		printf("%d ", seq[i]);
	}
	printf("\nedges: ");
	for(int i = 0; i < len; i++){
		for(int e = start[i]; e < start[i+1]; e++){
			printf("%d ", edge[e]);		
		}	
		printf("| ");
	}
	printf("\nendnodes: ");
	for(int i = 0; i< len; i++){
		printf("%u ", endlist[i]);
	}
	printf("\n");
}

template<int SL, int MAXL, int WL>
__global__ void fuse_lpo(int i_seq_idx, int j_seq_idx, int* nseq_offsets) {
	
	if(g_space_exceeded) return;

	int nseq;
	int block_offset;
	int myId = blockIdx.x;
	int y_seq_offs;
	int y_edge_offs;
	int len_y;
	int len_x;
	
	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = old_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = old_len_global[myId];
	}
	
	y_seq_offs = lpo_offsets[block_offset + j_seq_idx-1];
	y_edge_offs = lpo_edge_offsets[block_offset + j_seq_idx-1];
	len_y = lpo_offsets[block_offset + j_seq_idx] - lpo_offsets[block_offset + j_seq_idx-1];
	
	if(j_seq_idx < nseq){
		
		int global_seq_idx_y = j_seq_idx + block_offset;
		unsigned char* seq_x = dyn_letters_global + MAXL * myId;
		unsigned char* seq_y = lpo_letters + y_seq_offs;
		Edge* left_x = dyn_edges_global + MAXL * EDGE_F * myId;
		Edge* left_y = lpo_edges + y_edge_offs;
		int* start_x = dyn_edge_bounds_global + (MAXL+1) * myId;
		int* start_y = edge_bounds + y_seq_offs + global_seq_idx_y;
		unsigned char* end_list_x = dyn_end_nodes_global + MAXL * myId;
		unsigned char* end_list_y = end_nodes + y_seq_offs;
		int* x_to_y = x_to_ys + MAXL * myId;
		int* y_to_x = y_to_xs + MAXL * myId;
		unsigned char* seq_ids_x = dyn_sequence_ids_global + MAXL * WL * myId;
		unsigned char* seq_ids_y = sequence_ids + y_seq_offs * WL;

		unsigned char* new_seq = new_letters_global + MAXL * myId;
		Edge* new_left = new_edges_global + MAXL * EDGE_F * myId;
		int* new_start = new_edge_bounds_global + (MAXL+1) * myId;
		unsigned char* new_end_list = new_end_nodes_global + MAXL * myId;
		unsigned char* new_seq_ids = new_sequence_ids_global + MAXL * WL * myId;
		
		int x_to_y_idx = 0;
		int y_to_x_idx = 0;
		int i = 0;
		int j = 0;
		int n = 0;
		int edge_n = 0;
		unsigned char move_i = 0;
		unsigned char move_j = 0;
		int insertion_x[MAXL];
		int insertion_y[MAXL];
		int insertion_x_count = 0;
		int insertion_y_count = 0;
		
		while (i < len_x || j < len_y) {

			new_start[n] = edge_n; 
			
			if(j >= len_y){		
				
				int start_x_edges = start_x[i];
				int end_x_edges = start_x[i + 1];
				insertion_y[i] = insertion_y_count; 
				for (int e_i = start_x_edges; e_i < end_x_edges; e_i++, edge_n++) {
					new_left[edge_n] =  left_x[e_i] >= 0 ? left_x[e_i] + insertion_y[left_x[e_i]] : left_x[e_i];
				}
				new_seq[n] = seq_x[i];
				new_end_list[n] = end_list_x[i];	
				
				for (int k = 0; k < WL; k++) {
					unsigned char id_x = seq_ids_x[i*WL + k];
					new_seq_ids[n*WL + k] = id_x;
				}
				n++;
				i++;
				continue;

			} else if(i >= len_x ){
				int start_y_edges = start_y[j];
				int end_y_edges = start_y[j + 1];
				insertion_x[j] = insertion_x_count;
				for (int e_j = start_y_edges; e_j < end_y_edges; e_j++, edge_n++) {
					new_left[edge_n] = left_y[e_j] >= 0 ? left_y[e_j] + insertion_x[left_y[e_j]] : left_y[e_j];
				}
				new_seq[n] = seq_y[j];
				new_end_list[n] = end_list_y[j];
				
				for (int k = 0; k < WL; k++) {
					unsigned char id_y = seq_ids_y[j*WL + k];
					new_seq_ids[n*WL + k] = id_y;
				}
				n++;
				j++;
				continue;
			}

			move_i = 0;
			move_j = 0;
			x_to_y_idx = x_to_y[i];
			y_to_x_idx = y_to_x[j];
			
			int start_x_edges = start_x[i];
			int end_x_edges = start_x[i + 1];
			int start_y_edges = start_y[j];
			int end_y_edges = start_y[j + 1];
			int start_edge_n = edge_n;

			if (x_to_y_idx == NOT_ALIGNED) {

				for (int e_i = start_x_edges; e_i < end_x_edges; e_i++, edge_n++) {
					new_left[edge_n] =  left_x[e_i] >= 0 ? left_x[e_i] + insertion_y[left_x[e_i]] :  left_x[e_i];
				}
				new_seq[n] = seq_x[i];
				new_end_list[n] = end_list_x[i];
				insertion_x_count++;
				move_i = 1;
			}
			else if (y_to_x_idx == NOT_ALIGNED) {

				for (int e_j = start_y_edges; e_j < end_y_edges; e_j++, edge_n++) {
					new_left[edge_n] = left_y[e_j] >= 0 ? left_y[e_j] + insertion_x[left_y[e_j]] : left_y[e_j];
				}
				new_seq[n] = seq_y[j];
				new_end_list[n] = end_list_y[j];
				insertion_y_count++;
				move_j = 1;
			}
			else if (seq_x[y_to_x_idx] == seq_y[x_to_y_idx]) {

				for (int e_i = start_x_edges; e_i < end_x_edges; e_i++, edge_n++) {
					new_left[edge_n] = left_x[e_i] >= 0 ? left_x[e_i] + insertion_y[left_x[e_i]] : left_x[e_i];
				}
				for (int e_j = start_y_edges; e_j < end_y_edges; e_j++) {
					
					unsigned char positive_pred = left_y[e_j] >= 0;
					int jpos = positive_pred ? left_y[e_j] + insertion_x[left_y[e_j]] : left_y[e_j];
					unsigned char already_inserted = 0;
					
					for (int e_i = 0; e_i < end_x_edges - start_x_edges; e_i++) {
						if (new_left[start_edge_n + e_i] == jpos) {
							already_inserted = 1;
							break;
						}
					}
					if (already_inserted == 0) {
						if(positive_pred == 0 && end_y_edges-start_y_edges==1){
							int n_x_edges = end_x_edges-start_x_edges;
							int edge_start = edge_n - (n_x_edges);
							for(int sh = n_x_edges+edge_start-1; sh >= edge_start; sh--){
								new_left[sh+1] = new_left[sh];
							}
							new_left[edge_start] = left_y[e_j];
							edge_n++;
						}else{
							new_left[edge_n] = jpos;
							edge_n++;
						}
					}
				}
				new_seq[n] = seq_x[i];
				new_end_list[n] = (end_list_x[i] != 0) && (end_list_y[j] != 0);

				move_i = 1;
				move_j = 1;

			}
			
			insertion_x[j] = insertion_x_count;
			insertion_y[i] = insertion_y_count;

			for (int k = 0; k < WL; k++) {
				unsigned char id_x = (move_i == 0) ? 0 : seq_ids_x[i*WL + k];
				unsigned char id_y = (move_j == 0) ? 0 : seq_ids_y[j*WL + k];
				new_seq_ids[n*WL + k] = id_x + id_y;
			}

			if (move_i) { i++; }
			if (move_j) { j++; }
			n++;
		}
		new_start[n] = edge_n;
	}
}


__device__ inline unsigned char bp_map(char c) {

	if (c == 'A') {
		return 0;
	}
	if (c == 'C') {
		return 4;
	}
	if (c == 'G') {
		return 7;
	}
	if (c == 'T') {
		return 16;
	}
	return 8;
}
__device__ inline char bp_reverse(int i) {

	if (i == 0) {
		return 'A';
	}
	if (i == 4) {
		return 'C';
	}
	if (i == 7) {
		return 'G';
	}
	if (i == 16) {
		return 'T';
	}
	return 'N';
}

template<int SL, int MAXL, int WL>
__global__ void compute_edge_offsets(int* seq_offsets, int* nseq_offsets){
	
	int myId = blockIdx.x;	
	int seq_idx = threadIdx.x;
	int block_offset;
	int n_seq;
	
	lpo_offsets = seq_offsets; 
	
	if(myId == 0){
		block_offset = 0;
		n_seq = nseq_offsets[myId];

	}else{
		block_offset = nseq_offsets[myId-1];
		n_seq = nseq_offsets[myId] - nseq_offsets[myId-1];
	}
	
	if(seq_idx < n_seq){	
		lpo_edge_offsets[block_offset + seq_idx] = seq_offsets[block_offset + seq_idx];
	}
}

template<int SL, int MAXL, int WL>
__global__ void generate_lpo(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx) {

	int myTId = threadIdx.x;
	int myId = blockIdx.x;
	
	int char_idx = myTId;
	int seq_len;
	int offset;
	int edge_offset;
	int block_offset;
	int n_seq;
	
	if(myId == 0){
		block_offset = 0;
		n_seq = nseq_offsets[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		n_seq = nseq_offsets[myId] - nseq_offsets[myId-1];
	}
	
	if(block_offset + seq_idx == 0){ 
		offset = 0;                             
		edge_offset = 0;
		seq_len = seq_offsets[block_offset + seq_idx];
	}else{
		seq_len = seq_offsets[block_offset + seq_idx] - seq_offsets[block_offset + seq_idx-1];
		offset = seq_offsets[block_offset + seq_idx-1];
		edge_offset = lpo_edge_offsets[block_offset + seq_idx-1];
	}

	char* sequence = seq + offset; 
	unsigned char* seq_x;
	Edge* left_x;
	unsigned char* end_list_x;
	int* start_x;
	unsigned char* seq_ids;

	if(seq_idx == 0){
		dyn_len_global[myId] = seq_len;
		seq_x = dyn_letters_global + MAXL * myId;
		left_x = dyn_edges_global + MAXL * EDGE_F * myId;
		end_list_x = dyn_end_nodes_global + MAXL * myId;
		start_x = dyn_edge_bounds_global + (MAXL+1) * myId;
		seq_ids = dyn_sequence_ids_global + MAXL * WL * myId;
	}else{
		seq_x = lpo_letters + offset;
		left_x = lpo_edges + edge_offset;
		end_list_x = end_nodes + offset;
		start_x = edge_bounds + offset + seq_idx + block_offset;
		seq_ids = sequence_ids + offset*WL;
	}
	
	if (char_idx < seq_len && seq_idx < n_seq) {
		
		unsigned char not_end = char_idx != seq_len - 1;
		seq_x[char_idx] = bp_map(sequence[char_idx]);
		end_list_x[char_idx] = not_end;
		left_x[char_idx] = char_idx - 1;
		for(int id = 0; id < WL; id++)
			seq_ids[(char_idx*WL) + id] = id == seq_idx;
		start_x[char_idx] = char_idx;
	}
	if(char_idx == seq_len && seq_idx < n_seq) {
		start_x[char_idx] = char_idx;
	}
}

template<int SL, int MAXL, int WL>
__global__ void copy_result_sizes(int *nseq_offsets, int* res_size){

	int myId = blockIdx.x;
	int nseq;
	if(myId == 0){
		nseq = nseq_offsets[myId];
		free(moves_y_global);
	}else{
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
	}
	
	res_size[myId] = dyn_len_global[myId] * nseq;
}

template<int SL, int MAXL, int WL>
__global__ void compute_result(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx) {

	int myId = blockIdx.x;
	int char_idx = threadIdx.x % MAXL;
	int res_offset;
	int nseq;
	int len_x;
	
	if(myId == 0){
		nseq = nseq_offsets[myId];
		res_offset = 0;
		len_x = dyn_len_global[myId];
	}else{
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		res_offset = seq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}
	
	unsigned char* seq_x = dyn_letters_global + MAXL * myId;
	unsigned char* seq_x_ids = dyn_sequence_ids_global + MAXL * WL * myId;
	char* my_result = result + res_offset;
	
	if(seq_idx < nseq && char_idx < len_x) {
		my_result[seq_idx*len_x + char_idx] = seq_x_ids[char_idx*WL + seq_idx] ? bp_reverse(seq_x[char_idx]) : '-';
	}
}

template<int SL, int MAXL, int WL>
__global__ void compute_d_offsets(int i_seq_idx, int j_seq_idx, int* nseq_offsets) {
	
	if(g_space_exceeded) return;

	int nseq;
	int block_offset;
	int myId = blockIdx.x;
	int len_x;
	int len_y;

	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = dyn_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}
	if(j_seq_idx < nseq){
		
		len_y = lpo_offsets[block_offset + j_seq_idx] - lpo_offsets[block_offset + j_seq_idx-1];

		int* d_offsets = d_offsets_global + (MAXL + SL+1) * myId;  

		int min_d = len_x < len_y ? len_x : len_y;
		int offset = 0;
		for (int j = 0; j < len_x + len_y + 1; j++) {
			int n = min_d + 1 < j + 1 ? min_d + 1 : j + 1;
			n = n < len_x + len_y + 1 - j ? n : len_x + len_y + 1 - j;
			d_offsets[j] = offset;
			offset += n;
		}
	}
}

template<int SL, int MAXL, int WL>
__global__ void compute_new_lpo_size(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded) {

	if(g_space_exceeded) return;

	int nseq;
	int block_offset;
	int myId = blockIdx.x;
	int len_x;
	int len_y;
	
	if(myId == 0){
		block_offset = 0;
		nseq = nseq_offsets[myId];
		len_x = dyn_len_global[myId];
	}else{
		block_offset = nseq_offsets[myId-1];
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
		len_x = dyn_len_global[myId];
	}
	len_y = lpo_offsets[block_offset + j_seq_idx] - lpo_offsets[block_offset + j_seq_idx-1];
	
	if(j_seq_idx < nseq){

		int* y_to_x = y_to_xs + MAXL * myId;
		
		int count_unmapped = 0;
		for (int j = 0; j < len_y; j++) {
			count_unmapped += (y_to_x[j] == NOT_ALIGNED) ? 1 : 0;
		}
		old_len_global[myId] = dyn_len_global[myId];
		dyn_len_global[myId] = len_x + count_unmapped;
		if(len_x + count_unmapped >= MAXL){
			printf("ERROR: length %d exceeded maximmum limit(%d)\n", len_x + count_unmapped, MAXL);
			*space_exceeded = 1;
			g_space_exceeded = 1;
		}
	}
}

template<int SL, int MAXL, int WL>
__global__ void copy_new_lpo_data(int j_seq_idx, int* nseq_offsets) {
	
	if(g_space_exceeded) return;

	int myTId = threadIdx.x;	
	int nseq;
	int myId = blockIdx.x;
	int len_x;
	
	if(myId == 0){
		nseq = nseq_offsets[myId];
	}else{
		nseq = nseq_offsets[myId] - nseq_offsets[myId-1];
	}
	
	len_x = dyn_len_global[myId];
	
	unsigned char* seq_x = dyn_letters_global + MAXL * myId;
	Edge* left_x = dyn_edges_global + MAXL * EDGE_F * myId;
	int* lx_start = dyn_edge_bounds_global + (MAXL+1) * myId;
	unsigned char* end_list_x = dyn_end_nodes_global + MAXL * myId;
	unsigned char* seq_ids_x = dyn_sequence_ids_global + MAXL * WL * myId;

	if(j_seq_idx < nseq && myTId < len_x){
		
		seq_x[myTId] = new_letters_global[MAXL * myId + myTId];
		for(int k = 0; k < EDGE_F; k++){ 
			left_x[EDGE_F * myTId + k] = new_edges_global[MAXL * EDGE_F * myId + EDGE_F * myTId + k]; 
		}
		lx_start[myTId] = new_edge_bounds_global[(MAXL+1) * myId + myTId];
		end_list_x[myTId] = new_end_nodes_global[MAXL * myId + myTId];
		for(int k = 0; k < WL; k++){ 
			seq_ids_x[WL * myTId + k] = new_sequence_ids_global[MAXL * WL * myId + WL * myTId + k]; 
		}
	}
	if(j_seq_idx < nseq && myTId == len_x){
		lx_start[myTId] = new_edge_bounds_global[(MAXL+1) * myId + myTId];
	}
}

//The file below contains the template specializations for the above cuda kernels. Edit this file to add new template specializations.
#include "templates_impl.cpp"
