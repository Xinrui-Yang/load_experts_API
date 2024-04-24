main: load_experts_cuda.cu
	nvcc -o main load_experts_cuda.cu 

run: main
	./main

clean:
	rm -rf main

all:
	make clean; make; make run

ncu: 
	make clean; make; ncu -o report --import-source 1 --set full -f main