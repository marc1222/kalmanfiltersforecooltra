#include "eigen/Eigen/Dense">
#include <iostream>
using namespace Eigen;

int main() {
	MatrixXd x(2,1);
	x(0,0) = 2;
	x(1,0) = 4;
	std::cout << x << std::endl;
	MatrixXd C(1,2);
	C(0,0) = 2;
	C(0,1) = 4;
	std::cout << C << std::endl;
	MatrixXd aux = (C*x);
	double a = double(aux(0,0));
	std::cout << a << std::endl;
}
