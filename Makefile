all:
	g++ -std=c++17 -O3 main.cpp -o cpp_grad.o

clean:
	rm -r *.o
