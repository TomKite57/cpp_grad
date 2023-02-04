all:
	g++ -std=c++20 main.cpp -o cpp_grad.o

clean:
	rm -r *.o