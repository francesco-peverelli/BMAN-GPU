# CC=/usr/bin/g++
CXX= g++
CXX_GPU= nvcc
NVCC_W= -w
CFLAGS = -O3 -std=c++11 -lpthread  -IBOA -IBOA_GPU
LFLAGS =-IBOA -IBOA_GPU 
NVCC_LFLAGS= -lcudadevrt
EXEC=testGPU
OBJS := *.o BOA_GPU/workers.o BOA_GPU/cuda-poa.o BOA_GPU/poa.o BOA/align_lpo2.o  BOA/align_lpo_po2.o  BOA/align_score.o  BOA/black_flag.o  BOA/buildup_lpo.o  BOA/create_seq.o  BOA/fasta_format.o  BOA/heaviest_bundle.o  BOA/lpo_format.o  BOA/lpo.o   BOA/msa_format.o  BOA/numeric_data.o  BOA/remove_bundle.o  BOA/seq_util.o  BOA/stringptr.o Complete-Striped-Smith-Waterman-Library/src/*.o

all: $(EXEC)

ifeq ($(prof),1)
 CFLAGS+= -pg
endif
ifeq ($(deb),1)
 CFLAGS+= -O0 -DASSERTS -g
endif

ifeq ($(sani),1)
 CFLAGS= -std=c++11 -lpthread -fsanitize=address -fno-omit-frame-pointer -O1 -fno-optimize-sibling-calls -g
endif



#test:
#	./testLR
#
all: $(EXEC)



#testLR:  testLR.o bmean.o utils.o
#	$(CXX_GPU) $(NVCC_W) $(LFLAGS)  $(OBJS) -o $@

lpo_test: bmean.o utils.o bmean_test.o input_utils.o lpo_test.o
ifneq (,$(wildcard ./gpu_poa_test.o))
	    rm gpu_poa_test.o
endif
	$(CXX_GPU) $(NVCC_LFLAGS)  $(NVCC_W) $(LFLAGS)  $(OBJS) -o $@

testGPU: bmean.o utils.o bmean_test.o input_utils.o gpu_poa_test.o 
ifneq (,$(wildcard ./lpo_test.o))
	    rm lpo_test.o
endif
	$(CXX_GPU) $(NVCC_LFLAGS)  $(NVCC_W) $(LFLAGS)  $(OBJS) -o $@ 

%.o: %.cpp
	$(CXX_GPU) $(NVCC_W) -o $@ -x cu -c $< $(CFLAGS)


clean:
	rm -rf *.o
ifneq (,$(wildcard ./testGPU))
	rm testGPU
endif
ifneq (,$(wildcard ./lpo_test))
	rm lpo_test
endif



rebuild: clean $(EXEC)
