#include <cuda-runtime.h>

struct localisation {
	uint32_t read_id;
	int32_t position;
};

struct score_chain {
	int32_t length;
	int32_t score;
	int32_t next_anchor;
};

__host__ void anchor_scoring( int* S, size_t r_size, localisation* loc_v, size_t loc_size, int* loc_offsets ){

	dim3 grid_size( r_size, r_size );

	if(r_size > 1024){
	       	cout << "Max size exceeded" << endl;
		exit(-1);
	}

	int *S_d;
	localisation* loc_v_d;
	int* loc_offsets_d;

	cudaMalloc(&S_d, r_size * r_size * sizeof(int));
	cudaMalloc(&loc_v_d, loc_size * sizeof(struct localisation));
	cudaMalloc(&loc_offsets_d, r_size * sizeof(int));

	cudaMemcpy(loc_v_d, loc_v, loc_size * sizeof(struct localisation));
	cudaMemcpy(loc_offsets_d, r_size * sizeof(int));
}
