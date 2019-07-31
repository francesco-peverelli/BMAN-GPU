COMPILER = g++
COMPILER_GPU = nvcc
CC = gcc
CFLAGS = -O3 -std=c++11 -lpthread
INCLUDE_CUDA = -I /usr/local/cuda-10.1/include/ 

#BMEAN_LIB = $(BMEAN_DIR)/bmean.o $(BMEAN_DIR)/utils.o 
#BOA_LIB = $(BOA_DIR)/align_lpo2.o  $(BOA_DIR)/align_lpo_po2.o  $(BOA_DIR)/align_score.o  $(BOA_DIR)/black_flag.o  $(BOA_DIR)/buildup_lpo.o  $(BOA_DIR)/create_seq.o  $(BOA_DIR)/fasta_format.o  $(BOA_DIR)/heaviest_bundle.o  $(BOA_DIR)/lpo_format.o  $(BOA_DIR)/lpo.o   $(BOA_DIR)/msa_format.o  $(BOA_DIR)/numeric_data.o  $(BOA_DIR)/remove_bundle.o  $(BOA_DIR)/seq_util.o  $(BOA_DIR)/stringptr.o

gpu: cuda-poa.cu

	$(COMPILER_GPU) --default-stream per-thread -lineinfo -dc cuda-poa.cu -o cuda-poa.o
	$(COMPILER_GPU) -lcudadevrt --default-stream per-thread -lineinfo -x cu -c workers.cpp -o workers.o 
	$(COMPILER_GPU) --default-stream per-thread -lineinfo -dlink cuda-poa.o workers.o -o poa.o -lcudadevrt
	#$(COMPILER_GPU) $(CFLAGS) --default-stream per-thread -I. -I $(BMEAN_DIR) -I $(BOA_DIR) -lineinfo -x cu -c lpo_fusion.cpp -o test.o  	
	#$(COMPILER_GPU) $(CFLAGS) --default-stream per-thread -lineinfo link.o $(BOA_LIB) $(BMEAN_LIB) bmean_link.o cuda-poa.o test.o -o poa -lcudart

#bmean:

#	cd BMEAN && ./install.sh

clean:
#	cd BMEAN && ./clean.sh
	rm -f *.o
	rm -f error_file_archive.txt

all:
	
