#ifndef POA_CONSTANTS_H
#define POA_CONSTANTS_H

#define MATCH 1
#define MISMATCH -2
#define GAP 1
#define MAX_GAPL 16
#define USES_GLOBAL 1

#define BATCH_SIZE 1200000

#define WLEN_32 8
#define SEQ_LEN_32 31
#define MAX_SEQ_32 128
#define BLOCK_DIM_32 150000

#define WLEN_64 8
#define SEQ_LEN_64 63
#define MAX_SEQ_64 192
#define BLOCK_DIM_64 120000

#define WLEN_128 8
#define SEQ_LEN_128 127
#define MAX_SEQ_128 352
#define BLOCK_DIM_128 37984//40000

#define WLEN_255 8
#define SEQ_LEN_255 255
#define MAX_SEQ_255 830
#define BLOCK_DIM_255 5000//10910

#define WLEN_16_32 16
#define SEQ_LEN_16_32 31
#define MAX_SEQ_16_32 128
#define BLOCK_DIM_16_32 150000

#define WLEN_16_64 16
#define SEQ_LEN_16_64 63
#define MAX_SEQ_16_64 352
#define BLOCK_DIM_16_64 60000

#define WLEN_16_128 16	
#define SEQ_LEN_16_128 127
#define MAX_SEQ_16_128 448
#define BLOCK_DIM_16_128 28000

#define WLEN_16_255 16
#define SEQ_LEN_16_255 255
#define MAX_SEQ_16_255 960
#define BLOCK_DIM_16_255 7000

#define WLEN_32_32 32
#define SEQ_LEN_32_32 31
#define MAX_SEQ_32_32 352//256
#define BLOCK_DIM_32_32 50000//80000

#define WLEN_32_64 32
#define SEQ_LEN_32_64 63
#define MAX_SEQ_32_64 352
#define BLOCK_DIM_32_64 50000

#define WLEN_32_128 32	
#define SEQ_LEN_32_128 127
#define MAX_SEQ_32_128 512
#define BLOCK_DIM_32_128 20000

#define WLEN_32_255 32
#define SEQ_LEN_32_255 255
#define MAX_SEQ_32_255 960
#define BLOCK_DIM_32_255 5000

#define EDGE_F 3
#define N_THREADS 64

#define MIN_SLEN 1
#define MIN_WLEN 2

#define wrapSize 32
#define FULL_MASK 0xffffffff
#define D_BUFF_DEPTH 15


#endif //POA_CONSTANTS_H 
