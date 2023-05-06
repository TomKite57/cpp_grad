all:
	g++ -std=c++17 main.cpp -o cpp_grad.o

clean:
	rm -r *.o
