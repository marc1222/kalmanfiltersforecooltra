#include <iostream>
#include <math.h>
#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/MatrixFunctions"
#include <vector>

using namespace Eigen;


/* * 					
* * * ALGORITHM CONSTANTS  * * *
* */
#define NOMINAL_CAPACITY 20 //NOMINAL CAPACITY = RATED MANUFACTURER CAPACITY (Ah) ->  20AH
#define Euler_Aprox 2.71828182845904523536028747
#define total_iterations 70
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

/* * 					
* * * FUNCTION DECLARATION  * * *
* */
double compute_SOC(int t);
void compute_A();
void compute_B();
double compute_I(int t);
double compute_Vt();
double compute_Vp();

int get_soc_interval();
void assign_C_and_D();

double compute_lineal(double var[], double k1[], double k2[]);
void compute_intervals_vars();

double innovation_sum_squared(const std::vector<double>& sum);

void initialize_ekf();
void init_model_arrays();
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
double SOC;
int SOC_interval = 0;

//coeficients && values for SOC intervals (20 intervals -> interval[0] == interval[1], interval maps [i*5 - (i+1)*5] SOC
//Voc (OVC) array
double Voc[21], a[20], b[20];
//Rt (Ohmic resistance) array
double Rt[21], c[20], d[20];
//Rp (Diffusion resistance) array
double Rp[21],e[20], f[20]
//Cp (capacitance) array
double Cp[21], g[20], h[20];

//variables to save iteration values
double Voci, Rti, Rpi, Cpi;

//Vt voltage
double y;

//read & save actual current measure
MatrixXd u(1,1);


//measurement array
double current_measurements[total_iterations];

int main() {
	/* * 					
	* * * INITALIZATION OF ALGORITHM  * * *
	* */
	init_model_arrays(); //init model arrays
    std::vector<double> e(2);
    e[0] = 0;
	MatrixXd K; //Kalman Gain
	MatrixXd H(1,1);
	initialize_ekf(); //initialize SOC_vars, SOC[0], x[k-1], P[k-1], Q[k-1], R[k-1], A[k-1], B[k-1], , u[k-1] current Model measurements depending on SOC (initialized at 100%)
	/* * 					
	* * * BEGIN ITERATIVE ALGORITHM  * * *
	* */
	iteration = 1;
	for (int i = 0; i < total_iterations; ++i) {
		//2 - priori process
		x = A*x + B*u(0,0); //X[k] computation
		P = A*P*A.transpose()+Q; //P[k] computation
		//-------------------------------------------
		//3 - compute the innovation matrix and the Kalman gain
		//3.1 first compute new iteration values Vt[k] (y), C[k], D[k], u[k]
		u(0,0) = current_measurements[iteration]; //compute u[k]
		y = compute_Vt(); //compute y[k]
		assign_C_and_D(); //compute c[k] & d[k]
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
	SOC = 100; //Battery fully loaded
	SOC_interval = get_soc_interval();
	compute_intervals_vars();
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

//return soc interval of SOC global variable, 19 for 100%-95% and decreasing for each interval
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
	
	Voci = Voc[SOC_interval];
	Rti = Rt[SOC_interval];
	Rpi = Rp[SOC_interval];
	Cpi = Cp[SOC_interval];
	
}
//computes terminal voltage Vt in function of time -> Vt = Voci + Rti*I + Vp (Rpi*I)
double compute_vt() {
	double vt = Voci + Rti*u(0,0)+compute_Vp();
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

//computes capacitor voltage differential
//VP = Rtp*I
double compute_Vp() {
	
	return (Rpi*u(0,0));
	
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
	
	Voc[0] = 20.5;
	Voc[1] = 21;
	Voc[2] = 21.5;
	Voc[3] = 22;
	Voc[4] = 22.5;
	Voc[5] = 23;
	Voc[6] = 23.5;
	Voc[7] = 24;
	Voc[8] = 24.5;
	Voc[9] = 25;
	Voc[10] = 25.5;
	Voc[11] = 26;
	Voc[12] = 26.5;
	Voc[13] = 27;
	Voc[14] = 27.5;
	Voc[15] = 28;
	Voc[16] = 28.5;
	Voc[17] = 29;
	Voc[18] = 29.5;
	Voc[19] = 30;
	Voc[20] = 30.5;
	
	Rt[0] = 20.5;
	Rt[1] = 21;
	Rt[2] = 21.5;
	Rt[3] = 22;
	Rt[4] = 22.5;
	Rt[5] = 23;
	Rt[6] = 23.5;
	Rt[7] = 24;
	Rt[8] = 24.5;
	Rt[9] = 25;
	Rt[10] = 25.5;
	Rt[11] = 26;
	Rt[12] = 26.5;
	Rt[13] = 27;
	Rt[14] = 27.5;
	Rt[15] = 28;
	Rt[16] = 28.5;
	Rt[17] = 29;
	Rt[18] = 29.5;
	Rt[19] = 30;
	Rt[20] = 30.5;
	
	Rp[0] = 20.5;
	Rp[1] = 21;
	Rp[2] = 21.5;
	Rp[3] = 22;
	Rp[4] = 22.5;
	Rp[5] = 23;
	Rp[6] = 23.5;
	Rp[7] = 24;
	Rp[8] = 24.5;
	Rp[9] = 25;
	Rp[10] = 25.5;
	Rp[11] = 26;
	Rp[12] = 26.5;
	Rp[13] = 27;
	Rp[14] = 27.5;
	Rp[15] = 28;
	Rp[16] = 28.5;
	Rp[17] = 29;
	Rp[18] = 29.5;
	Rp[19] = 30;
	Rp[20] = 30.5;
	
	Cp[0] = 20.5;
	Cp[1] = 21;
	Cp[2] = 21.5;
	Cp[3] = 22;
	Cp[4] = 22.5;
	Cp[5] = 23;
	Cp[6] = 23.5;
	Cp[7] = 24;
	Cp[8] = 24.5;
	Cp[9] = 25;
	Cp[10] = 25.5;
	Cp[11] = 26;
	Cp[12] = 26.5;
	Cp[13] = 27;
	Cp[14] = 27.5;
	Cp[15] = 28;
	Cp[16] = 28.5;
	Cp[17] = 29;
	Cp[18] = 29.5;
	Cp[19] = 30;
	Cp[20] = 30.5;
}

void init_measurements() {
	
	current_measurements[0] = 0.1;
	current_measurements[1] = 0.3;
	current_measurements[2] = 0.5;
	current_measurements[3] = 0.6;
	current_measurements[4] = 0.67;
	current_measurements[5] = 0.2;
	current_measurements[6] = 0.4;
	current_measurements[7] = 0.56;
	current_measurements[8] = 0.54;
	current_measurements[9] = 0.4;
	current_measurements[10] = 0.45;
	current_measurements[11] = 0.76;
	current_measurements[12] = 0.67;
	current_measurements[13] = 0.73;
	current_measurements[14] = 0.43;
	current_measurements[15] = 0.12;
	current_measurements[16] = 0.16;
	current_measurements[17] = 0.21;
	current_measurements[18] = 0.31;
	current_measurements[19] = 0.31;
	current_measurements[20] = 0.41;
	current_measurements[21] = 0.45;
	current_measurements[22] = 0.54;
	current_measurements[23] = 0.33;
	current_measurements[24] = 0.32;
	current_measurements[25] = 0.19;
	current_measurements[26] = 0.25;
	current_measurements[27] = 0.34;
	current_measurements[28] = 0.69;
	current_measurements[29] = 0.51;
	current_measurements[30] = 0.74;
	current_measurements[31] = 0.53;
	current_measurements[32] = 0.672;
	current_measurements[33] = 0.6;
	current_measurements[34] = 0.45;
	current_measurements[35] = 0.56;
	current_measurements[36] = 0.23;
	current_measurements[37] = 0.45;
	current_measurements[38] = 0.59;
	current_measurements[39] = 0.45;
	current_measurements[40] = 0.31;
	current_measurements[41] = 0.34;
	current_measurements[42] = 0.41;
	current_measurements[43] = 0.29;
	current_measurements[44] = 0.46;
	current_measurements[45] = 0.33;
	current_measurements[46] = 0.23;
	current_measurements[47] = 0.16;	
	current_measurements[48] = 0.12;
	current_measurements[49] = 0.21;
	current_measurements[50] = 0.31;
	current_measurements[51] = 0.43;
	current_measurements[52] = 0.65;
	current_measurements[53] = 0.45;
	current_measurements[54] = 0.46;
	current_measurements[55] = 0.37;
	current_measurements[56] = 0.78;
	current_measurements[57] = 0.54;
	current_measurements[58] = 0.78;
	current_measurements[59] = 0.66;
	current_measurements[60] = 0.61;
	current_measurements[61] = 0.48;
	current_measurements[62] = 0.39;
	current_measurements[63] = 0.33;
	current_measurements[64] = 0.54;
	current_measurements[65] = 0.41;
	current_measurements[66] = 0.35;
	current_measurements[67] = 0.29;
	current_measurements[68] = 0.22;
	current_measurements[69] = 0.15;
	
}

void compute_coefficients (double & a[], double & b[], const double & measurements[] ) {
	for (int i = 1; i < 21; ++i) {
		dy = 
	}
}
