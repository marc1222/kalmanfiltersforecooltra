#include <iostream>
#include <math.h>
#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/MatrixFunctions"
#include <vector>

using namespace Eigen;


#define NOMINAL_CAPACITY 2.37 //NOMINAL CAPACITY = RATED MANUFACTURER CAPACITY (Ah)
#define Euler_Aprox 2.71828182845904523536028747
#define total_iterations 100
/*
Voc -> OVC
Rt -> ohmic resistence
Rp -> diffusion resistance
Cp -> diffusion capacitance

These values  above depend on SOC.
Voc (SOC) -> obtained from OVC test
Rt(SOC), Rp(SOC), Cp(SOC) obtained from DCIR test

Vt(t) -> circuit terminal voltage
I(t) -> load current (intensisty)

MATRIX MULTIPLICATION 
	1: The number of columns of the 1st matrix must equal the number of rows of the 2nd matrix. (A*B) = (Arows, Bcols)
	2: And the result will have the same number of rows as the 1st matrix, and the same number of columns as the 2nd matrix.
*/
//FUNCTION DECLARATIONS
//-------------------------------
double compute_SOC(int t);
void compute_A();
void compute_B();
double compute_I(int t);
double compute_Vt(int t);
double compute_Vp(int t);

int get_soc_interval();
void assign_C_and_D();

double compute_lineal(double var[], double k1[], double k2[]);
void compute_intervals_vars();

double innovation_sum_squared(const std::vector<double>& sum);

void initialize_ekf();
void init_model_arrays();
void init_measurement_arrays();
//-------------------------------
//SOC(t) = 1/NOMINAL CAPACITY * I(t) //I(t) -> read or estimated -> 4-5V --> SOC(t) = 1/Cn (nominal capacity) * I(t)
//Vp(t) = - 1/

int iteration;

MatrixXd A(2,2);
MatrixXd B(2,1); //2rows 1colum
MatrixXd x(2,1); // (SOC[k], Vp[k])transposed
MatrixXd C(1,2); //1 row 2 column (a 1)
MatrixXd Q(2,2);  //(2,2 matrix covariance)
MatrixXd P(2,2);

MatrixXd R(1,1); //(1,1 matrix covariance)
MatrixXd D(1,1); //(denotes Rt -> ohmic resistance)

int sampling_period = 2; //set to 2seconds
double SOC = -1;
int SOC_interval = 0;
//coeficients && values for SOC intervals (20 intervals -> interval[0] == interval[1], interval maps [i*5 - (i+1)*5] SOC
//a & b arrays are coeficients for Voc (OVC) array
double Voc[20], a[20], b[20];
//c & d arrays are coeficients for Rt (Ohmic resistance) array
double Rt[20], c[20], d[20];
//e & f arrays are coeficients for Rp (Diffusion resistance) array
double Rp[20], e[20], f[20];
//g & h arrays are coeficients for Cp (capacitance) array
double Cp[20], g[20], h[20];
//variables to save iteration values
double Voci, Rti, Rpi, Cpi;
//Vt voltage
double y;
//read actual current measure
MatrixXd u(1,1);


//measurement array
double current_measurements[total_iterations];

int main() {
	
	init_model_arrays();
	init_measurement_arrays();
    std::vector<double> e(2);
    e[0] = 0;
	MatrixXd K; //Kalman Gain
	MatrixXd H(1,1);
	initialize_ekf(); //initialize SOC_vars, SOC[0], x[k-1], P[k-1], Q[k-1], R[k-1], A[k-1], B[k-1], , u[k-1]
	iteration = 1;
	for (int i = 0; i < total_iterations; ++i) {
		//2 - priori process
		x = A*x + B*u(0,0); //X[k] computation
		P = A*P*A.transpose()+Q; //P[k] computation
		//3 - compute the innovation matrix and the Kalman gain
			//3.1 first compute new iteration values Vt[k] (y), C[k], D[k], u[k]
		y = compute_Vt(sampling_period*iteration); //compute y[k]
		assign_C_and_D(); //compute c[k] & d[k]
		u(0,0) = current_measurements[iteration]; //compute u[k]
		MatrixXd aux1 = (C*x);
		MatrixXd aux2 = (D*u);
		e[iteration] = y - ( aux1(0,0) + aux2(0,0) + b[SOC_interval]); //compute innovation matrix
		e.push_back(1); //make sure there is some free position
		MatrixXd aux3 = (C*P*C.transpose()+R).pow((-1));
		K = P*C.transpose()*aux3; //compute Kalman Gain [k]
		//4 - Adaptive covariance matching
		H(0,0) = 1/(iteration+1)*innovation_sum_squared(e);
		R = H - C*P*C.transpose();
		//5 - The posteriori process
		x = x + K*e[iteration];
		P = (u - K*C)*P;
		
		SOC = x(0,0); //get new soc
		SOC_interval = get_soc_interval(); //update interval
		compute_intervals_vars(); //update variable values
		compute_A(); //A[k]
		compute_B(); //B[k]
		iteration++;
	}
    return 0;
}

void initialize_ekf() {
	compute_intervals_vars();
	SOC = compute_SOC(0);
	SOC_interval = get_soc_interval();
	//compute Voc, Rt, Rp and Cp depending on soc interval
	//initialize x
	x(0,0) = SOC;
	x(1,0) = 0.04;
	//initialize P
	P(0,0) = 0.01;
	P(1,0) = 0;
	P(0,1) = 0;
	P(1,1) = 0.01;
	//initiallize R
	R(0,0) = 1;
	//initalize Q
	Q(0,0) = 1;
	Q(1,0) = 0;
	Q(0,1) = 0;
	Q(1,1) = 1;
	//initialize A
	compute_A();
	//compute B
	compute_B();
	//set first measurement
	u(0,0) = current_measurements[0];
}

//aplies SOC formula -> SOC(t) = 1/Cn (nominal capacity) * I(t)
double compute_SOC(int t) {
	 return (1/NOMINAL_CAPACITY*compute_I(t));
}

int get_soc_interval() {
	if (SOC > 95) return 19;
	else if (SOC > 90) return 18;
	else if (SOC > 85) return 17;
	else if (SOC > 80) return 16;
	else if (SOC > 75) return 15;
	else if (SOC > 70) return 14;
	else if (SOC > 65) return 13;
	else if (SOC > 60) return 12;
	else if (SOC > 55) return 11;
	else if (SOC > 50) return 10;
	else if (SOC > 45) return 9;
	else if (SOC > 40) return 8;
	else if (SOC > 35) return 7;
	else if (SOC > 30) return 6;
	else if (SOC > 25) return 5;
	else if (SOC > 20) return 4;
	else if (SOC > 15) return 3;
	else if (SOC > 10) return 2;
	else if (SOC > 5) return 1;
	else return 0;

}

void compute_A() {
	A(0,0) = 1;
	A(1,0) = 0;
	A(0,1) = 0;
	A(1,1) = 1-(sampling_period/(Rpi*Cpi));
}
void compute_B() {
	Matrix2d aux(1,2);
	aux(0,0) = sampling_period/NOMINAL_CAPACITY;
	aux(0,1) = sampling_period/Cpi;
	B = aux.transpose();
}
void assign_C_and_D() {
	C(0,0) = a[SOC_interval];
	C(0,1) = 1;
	
	D(0,0) = Rti;
}

//compues lineal equation on soc_interval -> var[soc_interval]+k1[soc_i]+k2[soc_i]
double compute_lineal(double var[], double k1[], double k2[]) {
	return (var[SOC_interval]*k1[SOC_interval]+k2[SOC_interval]);
}
//computes Voci, Rti, Rpi and Cpi
void compute_intervals_vars() {
	Voci = compute_lineal(Voc, a, b);
	Rti = compute_lineal(Rt, c, d);
	Rpi = compute_lineal(Rp, e, f);
	Cpi = compute_lineal(Cp, g, h);
}
//computes terminal voltage Vt in function of time
double compute_vt(int t) {
	double vt = Voci + Rti*compute_I(t)+compute_Vp(t);
	return vt;
}
//computes current in function of time
// i(t) = V/(rti+pi)*(1-exp(-alpha*t/Rti*Cpi))+(V/Rti)*exp(-alpha*t/Rti*Cpi) -> alpha = (Rti/Rpi) + 1
double compute_I(int t) {
	double aux1 = Voci/(Rti+Rpi);
	double alpha = Rti/Rpi + 1;
	double aux2 = pow(Euler_Aprox, ((-alpha*t)/(Rti*Cpi)));
	return (aux1*(1-aux2)+(Voci/Rti)*aux2);
}
//computes capacitor voltage in function of time
//Vp = Q(t)/Cpi
//Q(t) = CV*(1-exp(-alpha*t/Rti*Cpi))
double compute_Vp(int t) {
	double alpha = Rti/Rpi + 1;
	double aux = pow(Euler_Aprox, ((-alpha*t)/(Rti*Cpi)));
	double Q = Cpi*Voci*(1-aux);
	return (Q/Cpi);
}

double innovation_sum_squared(const std::vector<double>& sum) {
	double sumtotal = 0;
	double aux;
	for (int i = 1; i <= iteration; ++i) {
		aux = sum[i];
		sumtotal += (aux*aux);
	}
	return sumtotal;
}
//init model arrays
void init_model_arrays() {
	
}
//init measurement array
void init_measurement_arrays() {
	
}
