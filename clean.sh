cd Complete-Striped-Smith-Waterman-Library/src;
make clean;
cd ../..;
cd BOA;
make clean;
cd ..;
cd BOA_GPU
rm *.o
make gpu
cd ..;
make clean;
