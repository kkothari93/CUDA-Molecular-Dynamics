// Include files

#include <ctime>
#include "cuda_runtime.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include<string.h>

// Parameters

#define N_ATOMS 512
#define MASS_ATOM 1.0f
#define time_step 0.01f
#define L 8.5f
#define T 0.728f
#define NUM_STEPS 1000

const int BLOCK_SIZE = 1024;
const int BLOCK_SIZE1 = int(N_ATOMS / 2);
const int scheme = 1; // 0 for explicit, 1 for implicit

/*************************************************************************************************************/
/*************								INITIALIZATION CODE										**********/
/*************************************************************************************************************/
__global__ void init_r(float* r, int N_cube){
	int ix = threadIdx.x + blockDim.x* blockIdx.x;
	int iy = threadIdx.y + blockDim.y* blockIdx.y;
	int iz = threadIdx.z + blockDim.z* blockIdx.z;
	int index = ix + iy*N_cube + iz * N_cube * N_cube;
	if (ix < N_cube && iy < N_cube && iz<N_cube){
		r[index] = L / 2.0 * (1.0 - float(2 * ix + 1) / N_cube);
		r[index + N_ATOMS] = L / 2.0 * (1.0 - float(2 * iy + 1) / N_cube);
		r[index + 2 * N_ATOMS] = L / 2.0 * (1.0 - float(2 * iz + 1) / N_cube);
	}
}

void init_vel(float* vel){
	srand(time(NULL));
	float netP[3] = { 0.0f, 0.0f, 0.0f };
	float netE = 0.0;
	int i;
	float a, b, c;
	for (i = 0; i < N_ATOMS; i++){
		a = ((float)rand() / RAND_MAX) - 0.5;
		b = ((float)rand() / RAND_MAX) - 0.5;
		c = ((float)rand() / RAND_MAX) - 0.5;
		vel[i] = a; vel[i + N_ATOMS] = b; vel[i + 2 * N_ATOMS] = c;
		netE += a*a + b*b + c*c;
		netP[0] += a;
		netP[1] += b;
		netP[2] += c;
	}
	netP[0] *= 1.0f / N_ATOMS;
	netP[1] *= 1.0f / N_ATOMS;
	netP[2] *= 1.0f / N_ATOMS;
	float vscale = sqrtf(3 * N_ATOMS*T / netE);
	for (i = 0; i < N_ATOMS; i++){
		vel[i] = (vel[i] - netP[0])*vscale;
		vel[i + N_ATOMS] = (vel[i + N_ATOMS] - netP[1])*vscale;
		vel[i + 2 * N_ATOMS] = (vel[i + 2 * N_ATOMS] - netP[2])*vscale;


	}
	//netP[0] = 0.0f;
	//netP[1] = 0.0f;
	//netP[2] = 0.0f;
	//for (i = 0; i < N_ATOMS; i++){
	//netP[0] += vel[i];
	//netP[1] += vel[i + N_ATOMS];
	//netP[2] += vel[i + 2 * N_ATOMS];
	//}
	//printf("netP in x %f \n", netP[0]);
	//printf("netP in y %f \n", netP[1]);
	//printf("netP in z %f \n", netP[2]);
}

/*************************************************************************************************************/
/*************									COMPUTATION KERNELS									**********/
/*************************************************************************************************************/

__global__ void makePairs(float* positions, float* pairpos){
	__shared__ float pos[N_ATOMS * 3];
	float del;
	int iatom1, iatom2, i;
	int tx = threadIdx.x;
	int n = 0, breachindex = -1;
	int index = tx + blockDim.x*blockIdx.x;
	i = tx;
	while (i < N_ATOMS * 3){
		pos[i] = positions[i];
		i += blockDim.x;
	}
	__syncthreads();
	if (index < N_ATOMS*N_ATOMS){
		iatom1 = int(index / N_ATOMS);
		iatom2 = index - iatom1*N_ATOMS;
#pragma unroll
		for (i = 0; i < 3; i++){
			del = pos[iatom1 + i*N_ATOMS] - pos[iatom2 + i*N_ATOMS];
			if (fabs(del) > L / 2.0f) { del += (2 * (del<0) - 1)*L; }
			if (fabs(del)> L / 2.0f){ n = 1 + i; breachindex = index; }
			pairpos[index + i*N_ATOMS*N_ATOMS] = del;
		}
	}
	__syncthreads();
	//if (index == breachindex) { printf("Number of breaches: %i\n", n); }
	//if (index < N_ATOMS*N_ATOMS){
	//	del = sqrtf(__powf(pairpos[index], 2) + __powf(pairpos[index + N_ATOMS*N_ATOMS], 2) + __powf(pairpos[index + 2 * N_ATOMS*N_ATOMS], 2));
	//	if (del <1.0f && del != 0.0f){
	//	printf("distance breached at index %i\tatom 1: %i\tatom 2 : %i\tdistance: %f  (%f, %f, %f)\n", 
	//		index, iatom1, iatom2, del, pairpos[index], pairpos[index + N_ATOMS*N_ATOMS], pairpos[index + 2* N_ATOMS*N_ATOMS]);
	//	index = iatom2*N_ATOMS + iatom1;
	//	printf("Opposite distance at index %i: (%f, %f, %f)\n", index, pairpos[index], pairpos[index + N_ATOMS*N_ATOMS], pairpos[index + 2 * N_ATOMS*N_ATOMS]);
	//	}
	//}
}

__device__ float PutInBox(float r){
	if (fabs(r) > L / 2.0)
		r += (2 * (r < 0) - 1)*ceil((fabs(r) - L / 2.0f) / L)*L;
	return r;
}

__global__ void potForce(float * PairWise, int N, float * PotOut, float * ForceOut)
{
	/*
	PairWise - PairWise distances between atoms passed from global
	N - # atoms
	RowSize - # PairWise distances per block
	RowCumSize - # nonzero RowSize array elements = # blocks launched in parallel
	PotOut - Store the output Potential in global memory
	ForceOut - Store the output Force in global memory along x, 1D array size N*N
	*/
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	//Register variables to store pairwise separation
	float delx;
	float dely;
	float delz;
	float delr2, delrm6;
	float Potential;
	float Forcex;
	float Forcey;
	float Forcez;
	int row = tx + bx*BLOCK_SIZE;
	//if (row == 0) printf("I'm in 1! \n");
	if (row < N*N)
	{
		delx = PairWise[row];
		dely = PairWise[row + N*N];
		delz = PairWise[row + N*N * 2];
		delr2 = delx*delx + dely*dely + delz*delz;
		delrm6 = __powf(delr2, (float)-3);
		if (delr2 == 0.0) {
			Potential = 0;
			Forcex = 0;
			Forcey = 0;
			Forcez = 0;
		}
		else{
			Potential = 4 * __fadd_rn(delrm6*delrm6, -1 * delrm6);
			Forcex = -(delx / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
			Forcey = -(dely / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
			Forcez = -(delz / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
		}
		PotOut[row] = Potential;
		ForceOut[row] = Forcex;
		ForceOut[row + N*N] = Forcey;
		ForceOut[row + N*N * 2] = Forcez;
	}

}

/*************************************************************************************************************/
/*************							EXPLICIT KERNEL KINEMATICS									**********/
/*************************************************************************************************************/
__global__ void kinematics(float* positions, float* force, float* vel, int len){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int index = bx*blockDim.x + tx;
	float tempr;
	//if (index == 0){ printf("You have been trolled! \n"); }
	if (index < len){
		tempr = positions[index] + 0.5f * force[index] / MASS_ATOM * time_step*time_step + vel[index] * time_step;
		positions[index] = PutInBox(tempr);
		vel[index] += force[index] / MASS_ATOM * time_step;
	}
}

/*************************************************************************************************************/
/*************							IMPLICIT KERNEL KINEMATICS									**********/
/*************************************************************************************************************/
__global__ void kinematics_phase1(float* positions, float* force, float* vel, int len){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int index = bx*blockDim.x + tx;
	float tempr, tempa, tempvel;
	//if (index == 0){ printf("You have been trolled! \n"); }
	if (index < len){
		tempa = force[index] / MASS_ATOM;
		tempvel = vel[index];
		tempr = positions[index] + 0.5f * tempa * time_step*time_step + tempvel * time_step;
		positions[index] = PutInBox(tempr);
		vel[index] = tempvel + 0.5*tempa*time_step;
	}
}

__global__ void kinematics_phase2(float* force, float* vel, int len){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int index = bx*blockDim.x + tx;
	//if (index == 0){ printf("You have been trolled! \n"); }
	if (index < len){
		vel[index] += 0.5 * force[index] / MASS_ATOM * time_step;
	}
}

/*************************************************************************************************************/
/*************								REDUCTION  KERNELS										**********/
/*************************************************************************************************************/
__global__ void total(float *input, float *output, int len)
{
	//@@ Load a segment of the input vector into shared memory
	//@@ Traverse the reduction tree
	//@@ Write the computed sum of the block to the output vector at the
	//@@ correct index
	__shared__ float partSum[2 * BLOCK_SIZE];
	unsigned int tx = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
	//Loading input floats to shared memory
	//Take care of the boundary conditions 
	if (start + tx < len){
		partSum[tx] = input[start + tx];
		if (start + BLOCK_SIZE + tx <len) partSum[BLOCK_SIZE + tx] = input[start + BLOCK_SIZE + tx];
		else partSum[BLOCK_SIZE + tx] = 0;
	}
	else{
		partSum[tx] = 0;
		partSum[BLOCK_SIZE + tx] = 0;
	}
	unsigned int stride;
	for (stride = BLOCK_SIZE; stride > 0; stride = stride / 2){
		__syncthreads();
		if (tx < stride) partSum[tx] += partSum[tx + stride];
	}
	if (tx == 0){ output[blockIdx.x] = partSum[0]; }
}

__global__ void forcered_simple(float * force, float * forcered){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int i = 0;
	int findex;
	__shared__ float forcered_sh[3 * N_ATOMS];
	//if (index == 0){ printf("In force reduction kernel! \n"); }
	if (index < 3 * N_ATOMS){
		forcered_sh[index] = 0.0f;
	}
	__syncthreads();
	if (index < 3 * N_ATOMS){
		findex = int(index / N_ATOMS)*N_ATOMS*N_ATOMS + index % N_ATOMS;
		for (i = 0; i < N_ATOMS; i++){
			forcered_sh[index] += force[findex + i*N_ATOMS];
		}
	}
	__syncthreads();
	if (index < 3 * N_ATOMS){
		forcered[index] = forcered_sh[index];
	}
	/*if (index == 0){
	printf("forcered [0]= %f \n", forcered[0]);
	printf("forcered [2]= %f \n", forcered[2]);
	printf("forcered [4]= %f \n \n", forcered[4]);
	}*/
}

//__global__ void newForceReduction(float *input, float *output, int startunit, int len)
//{
//	unsigned int tx = threadIdx.x;
//	unsigned int start = blockIdx.x *N_ATOMS;
//
//	__shared__ float partSum[BLOCK_SIZE1];
//	// if (tx == 0) printf("Length of the shared memory array - %i \n", N_ATOMS);
//
//	//Loading input floats to shared memory
//	//Take care of the boundary conditions 
//	partSum[tx] = input[start + tx] + input[start + tx + BLOCK_SIZE1];
//	if (tx == 0){
//		if (N_ATOMS%2) partSum[0] += input[start + N_ATOMS - 1];
//	}
//	__syncthreads();
//
//	//Reduction Kernel for each dimension
//	unsigned int stride, stride1 = BLOCK_SIZE1;
//	for (stride = BLOCK_SIZE1/2; stride > 0; stride = stride / 2){
//		if (tx < stride) { partSum[tx] += partSum[tx + stride]; 
//		if (tx==0){ if (stride1%2) partSum[0] += partSum[stride1-1];}
//		}
//		__syncthreads();
//		stride1 = stride;
//	}
//	
//	if (tx == 0){
//		output[blockIdx.x] = -partSum[0];
//	}
//}

__global__ void newForceReduction(float *input, float *output, int startunit, int len)
{
	unsigned int tx = threadIdx.x;
	unsigned int start = blockIdx.x *N_ATOMS;

	__shared__ float partSum[BLOCK_SIZE];
	// if (tx == 0) printf("Length of the shared memory array - %i \n", N_ATOMS);

	//Loading input floats to shared memory
	//Take care of the boundary conditions
	if (tx < N_ATOMS) { partSum[tx] = input[start + tx]; }
	else{ partSum[tx] = 0.0f; }

	__syncthreads();

	//Reduction Kernel for each dimension
	if (tx < 512){
		partSum[tx] += partSum[tx + 512];
	} __syncthreads();
	if (tx < 256){
		partSum[tx] += partSum[tx + 256];
	} __syncthreads();
	if (tx < 128){
		partSum[tx] += partSum[tx + 128];
	} __syncthreads();
	if (tx < 64){
		partSum[tx] += partSum[tx + 64];
	} __syncthreads();
	if (tx < 32){
		partSum[tx] += partSum[tx + 32];
		partSum[tx] += partSum[tx + 16];
		partSum[tx] += partSum[tx + 8];
		partSum[tx] += partSum[tx + 4];
		partSum[tx] += partSum[tx + 2];
		partSum[tx] += partSum[tx + 1];
	}
	if (tx == 0){
		output[blockIdx.x] = -partSum[0];
	}
}

/*************************************************************************************************************/
/*************								HOST CODE GENERATING DATA								**********/
/*************************************************************************************************************/
void create_csv(char *filename, float *data, int Row, int Column){
	printf("\n Creating %s.csv file", filename);
	FILE *fp;
	int i, j;
	filename = strcat(filename, ".csv");
	fp = fopen(filename, "w+");
	//fprintf(fp,"Student Id, Physics, Chemistry, Maths");
	for (i = 0; i<Row; i++){
		//fprintf(fp,"%d",i+1);
		for (j = 0; j<Column; j++)
			//Assuming Row major way of storage
			fprintf(fp, ",%f", *(data + i*Column + j));
		fprintf(fp, "\n");
	}
	fclose(fp);
	printf("\n %sfile created \n", filename);
}

void create_dump(char *filename, float *data, int steps, int nparticles){
	printf("\n Creating %s.txt file", filename);
	FILE *fp;
	int i, j;
	filename = strcat(filename, ".txt");
	fp = fopen(filename, "w+");
	int stride;
	//fprintf(fp,"Student Id, Physics, Chemistry, Maths");
	/*ITEM: TIMESTEP
	0
	ITEM: NUMBER OF ATOMS
	130
	ITEM: BOX BOUNDS ff pp pp
	0 30
	0 30
	-0.5 0.5
	ITEM: ATOMS id type x y z ix iy iz*/
	for (i = 0; i<steps; i++){
		fprintf(fp, "ITEM: TIMESTEP \n%i\nITEM: NUMBER OF ATOMS\n%i\nITEM: BOX BOUNDS\n%f %f\n%f %f\n%f %f\nITEM: ATOMS id type x y z ix iy iz\n",
			i, N_ATOMS, -L / 2.0f, L / 2.0f, -L / 2.0f, L / 2.0f, -L / 2.0f, L / 2.0f);
		stride = i * 3 * N_ATOMS;
		for (j = 0; j<nparticles; j++)
			fprintf(fp, "%i %i %f %f %f 0 0 0\n", j + 1, j + 1, data[stride + j], data[stride + j + N_ATOMS], data[stride + j + 2 * N_ATOMS]);
	}
	fclose(fp);
	printf("\n %sfile created \n", filename);
}

/*************************************************************************************************************/
/*************									MAIN FUNCTION										**********/
/*************************************************************************************************************/
int main(){

	float * d_PotOut;
	float * d_ForceOut;
	float * d_PotRedOut;
	float * d_ForceOutRed;
	float * PotRedOut;
	float * ForceOut;
	float * PotOut;
	float * ForceOutRed;
	float* d_r;
	float* r;
	float* vel;
	float* d_vel;
	float* d_pairpos;
	float* pairpos;
	float* rpairpos;
	char str[100];
	float PEtrace[NUM_STEPS];
	float* R; //Output for OVITO
	const int size = sizeof(float) * N_ATOMS * 3;
	clock_t start, diff;
	cudaError_t state;
	// Declare space for storing positions and velocities on RAM
	r = (float *)malloc(size);
	vel = (float *)malloc(size);
	pairpos = (float *)malloc(size*N_ATOMS);
	rpairpos = (float *)malloc(size*N_ATOMS);
	ForceOut = (float *)malloc(size*N_ATOMS);
	PotOut = (float *)malloc(size*N_ATOMS / 3);
	PotRedOut = (float *)malloc(sizeof(float)*ceil(float(N_ATOMS * N_ATOMS) / BLOCK_SIZE)); //sizeof (float) needed to properly allocate memory in potredout
	ForceOutRed = (float *)malloc(size);
	R = (float *)malloc(size*NUM_STEPS); //put for OVITO
	// Declare space for storing positions and velocities on GPU DRAM
	cudaMalloc((void **)&d_r, size);
	cudaMalloc((void **)&d_vel, size);
	cudaMalloc((void **)&d_PotOut, size*N_ATOMS / 3);
	cudaMalloc((void **)&d_ForceOut, size*N_ATOMS);
	cudaMalloc((void **)&d_PotRedOut, sizeof(float)*ceil(float(N_ATOMS * N_ATOMS) / BLOCK_SIZE));
	cudaMalloc((void **)&d_ForceOutRed, size);
	cudaMalloc((void **)&d_pairpos, size*N_ATOMS);

	int N_cube = int(cbrt(float(N_ATOMS)));
	int gd = int(ceil(double(N_ATOMS) / N_cube));
	dim3 gridSize(1, 1, 1);
	dim3 blockSize(N_cube, N_cube, N_cube);
	init_r << <gridSize, blockSize >> >(d_r, N_cube);
	// check r
	//cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
	//for (int ii = 0; ii < N_ATOMS * 3; ii++){
	//printf("%f \t", r[ii]);
	//if ((ii + 1) % N_ATOMS == 0)
	//printf("\n");
	//}
	//printf("\n All r printed \n");
	//printf("Generation of r succeeded! \n");
	//char str[100];
	//printf("\n Enter the filename :");
	//gets(str);
	//create_csv(str, r , 3, N_ATOMS);

	// Initialise velocity
	init_vel(vel);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Init_r did not succeed: %s \n", cudaGetErrorString(state));
	}
	// Check velocity
	//for (int ii = 0; ii < N_ATOMS*3; ii++){
	//printf("%f \t", vel[ii]);
	//if ((ii+1) % N_ATOMS == 0)
	//printf("\n");
	//}
	//printf("\n All vel printed \n");

	// Copy velocity data to GPU
	cudaMemcpy(d_vel, vel, size, cudaMemcpyHostToDevice);


	for (int t = 0; t<NUM_STEPS; t++){
		// Create pairs
		if ((t == 0 && scheme == 1) || (scheme == 0)){
			makePairs << <ceil(N_ATOMS*N_ATOMS / float(BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_pairpos);
			state = cudaDeviceSynchronize();
			if (state != cudaSuccess){
				printf("make pairs did not succeed: %s \n", cudaGetErrorString(state));
			}
		}
		// printing r again when looping
		//cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
		//printf("\n Enter the filename :");
		//gets(str);
		//create_csv(str, r, 3, N_ATOMS);

		// print pair distances
		//state = cudaMemcpy(pairpos, d_pairpos, size*N_ATOMS, cudaMemcpyDeviceToHost);
		//if (state != cudaSuccess){
		//	printf("make pairs memcpy did not succeed: %s \n", cudaGetErrorString(state));
		//}
		//float distance;
		//for (int i = 0; i < N_ATOMS*N_ATOMS; i++){
		//	//rpairpos[i] = pairpos[i];
		//	distance = sqrtf(pairpos[i] * pairpos[i]
		//		+ pairpos[i + N_ATOMS*N_ATOMS] * pairpos[i + N_ATOMS*N_ATOMS]
		//		+ pairpos[i + 2 * N_ATOMS*N_ATOMS] * pairpos[i + 2 * N_ATOMS*N_ATOMS]);
		//	printf("%f \t", distance);
		//	if ((i + 1) % N_ATOMS == 0) printf("\n");
		//}
		//printf("\n All pair r printed \n");
		//printf("\n Enter the filename :");
		//gets(str);
		//create_csv(str, pairpos, N_ATOMS*3, N_ATOMS);


		int gridDim = ceil((N_ATOMS*N_ATOMS) / (float)BLOCK_SIZE);
		potForce << <gridDim, BLOCK_SIZE >> >(d_pairpos, N_ATOMS, d_PotOut, d_ForceOut);
		cudaDeviceSynchronize();
		// Check potential and force
		//cudaMemcpy(PotOut, d_PotOut, size*N_ATOMS / 3, cudaMemcpyDeviceToHost);
		//cudaMemcpy(ForceOut, d_ForceOut, size*N_ATOMS, cudaMemcpyDeviceToHost);
		//printf("\n Now printing non-reduced force array \n");
		//for (int ii = 0; ii < N_ATOMS*N_ATOMS * 3; ii++){
		//	printf("%f \t", ForceOut[ii]);
		//	if ((ii + 1) % N_ATOMS == 0)
		//		printf("\n");
		//}
		//printf("\n Enter the filename :");
		//gets(str);
		//create_csv(str, ForceOut , 3*N_ATOMS, N_ATOMS);


		total << < gridDim, BLOCK_SIZE >> >(d_PotOut, d_PotRedOut, N_ATOMS*N_ATOMS);
		//forcered_simple << <ceil(3 * N_ATOMS / float(BLOCK_SIZE)), BLOCK_SIZE >> >(d_ForceOut, d_ForceOutRed);
		newForceReduction << <3 * N_ATOMS, BLOCK_SIZE1 >> >(d_ForceOut, d_ForceOutRed, N_ATOMS, 3 * N_ATOMS*N_ATOMS);
		cudaDeviceSynchronize();
		int len = int(ceil(N_ATOMS*N_ATOMS / float(BLOCK_SIZE)));
		state = cudaMemcpyAsync(PotRedOut, d_PotRedOut, len*sizeof(float), cudaMemcpyDeviceToHost);

		if (state != cudaSuccess)
			printf("Reduction of potential memcpy failed %s \n", cudaGetErrorString(state));
		for (int i = 1; i < len; i++){
			PotRedOut[0] += PotRedOut[i];
		}
		//printf("Potential energy is %f\n", PotRedOut[0]/2.0f);
		PEtrace[t] = PotRedOut[0] / 2.0;

		//if (PotRedOut[0]>0.0){ printf("Simulation crashed!");  break; }
		if (t % 1000 == 0){ printf("Potential energy at iteration [%i] is %1.10f \n", t, PEtrace[t]); }
		
		//check force reduction
		//state = cudaMemcpy(ForceOutRed, d_ForceOutRed, size, cudaMemcpyDeviceToHost);
		//if (state != cudaSuccess)
		//	printf("Force memcpy failed: %s \n", cudaGetErrorString(state));
		//printf("Now printing reduced force array \n");
		//for (int ii = 0; ii < N_ATOMS * 3; ii++){
		//	printf("%f \t", ForceOutRed[ii]);
		//	if ((ii + 1) % N_ATOMS == 0)
		//		printf("\n");
		//}
		//char str[100];
		//printf("\n Enter the filename :");
		//gets(str);
		//create_csv(str, ForceOutRed , 3, N_ATOMS);
		
		//Explicit scheme
		if (scheme == 0){
			kinematics << < ceil((N_ATOMS * 3) / (float)BLOCK_SIZE), BLOCK_SIZE >> >(d_r, d_ForceOutRed, d_vel, 3 * N_ATOMS);
			cudaDeviceSynchronize();
			//state = cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
			//if (state != cudaSuccess)
			//	printf("r memcpy failed: %s \n", cudaGetErrorString(state));
			//printf("Now printing r after iteration %i \n", t);
			//for (int ii = 0; ii <3 * N_ATOMS; ii++){
			//	printf("%f \t", r[ii]);
			//	if ((ii + 1) % N_ATOMS == 0)
			//		printf("\n");
			//}
		}
		//Implicit scheme
		else{
			kinematics_phase1 << <ceil((N_ATOMS * 3) / (float)BLOCK_SIZE), BLOCK_SIZE >> >(d_r, d_ForceOutRed, d_vel, 3 * N_ATOMS);
			makePairs << <ceil(N_ATOMS*N_ATOMS / float(BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_pairpos);
			cudaDeviceSynchronize();
			potForce << <gridDim, BLOCK_SIZE >> >(d_pairpos, N_ATOMS, d_PotOut, d_ForceOut);
			cudaDeviceSynchronize();
			//forcered_simple << <ceil(3 * N_ATOMS / float(BLOCK_SIZE)), BLOCK_SIZE >> >(d_ForceOut, d_ForceOutRed);
			newForceReduction << <3 * N_ATOMS, BLOCK_SIZE >> >(d_ForceOut, d_ForceOutRed, N_ATOMS, 3 * N_ATOMS*N_ATOMS);
			state = cudaDeviceSynchronize();
			if (state != cudaSuccess)
				printf("Force kernel failed: %s \n", cudaGetErrorString(state));
			kinematics_phase2 << < ceil((N_ATOMS * 3) / (float)BLOCK_SIZE), BLOCK_SIZE >> >(d_ForceOutRed, d_vel, 3 * N_ATOMS);
			cudaDeviceSynchronize();
		}

		cudaMemcpyAsync(&R[0 + t * 3 * N_ATOMS], d_r, size, cudaMemcpyDeviceToHost);

	} // } for time integration for loop*/
	//printf("Enter name for creating positions dump: ");
	//gets(str);
	//create_dump(str, R, NUM_STEPS, N_ATOMS);
	// Print final pair distances
	//state = cudaMemcpy(pairpos, d_pairpos, size*N_ATOMS, cudaMemcpyDeviceToHost);
	//if (state != cudaSuccess){
	//	printf("make pairs memcpy did not succeed: %s \n", cudaGetErrorString(state));
	//}
	//float distance;
	//for (int i = 0; i < N_ATOMS*N_ATOMS; i++){
	//	//rpairpos[i] = pairpos[i];
	//	distance = sqrtf(pairpos[i] * pairpos[i]
	//		+ pairpos[i + N_ATOMS*N_ATOMS] * pairpos[i + N_ATOMS*N_ATOMS]
	//		+ pairpos[i + 2 * N_ATOMS*N_ATOMS] * pairpos[i + 2 * N_ATOMS*N_ATOMS]);
	//	printf("%f \t", distance);
	//	if ((i + 1) % N_ATOMS == 0) printf("\n");
	//}
	//printf("\n All pair r printed \n");
	//printf("\n Enter the filename :");
	//gets(str);
	//create_csv(str, PEtrace, 1, NUM_STEPS);
	//printf("\n Enter the filename :");
	//gets(str);
	//create_csv(str, r , 3, N_ATOMS);

	// Free up host memory
	free(r);
	free(vel);
	free(pairpos);
	free(rpairpos);
	//free(PEtrace);

	// Free up device memory
	cudaFree(d_ForceOut);
	cudaFree(d_ForceOutRed);
	cudaFree(d_pairpos);
	cudaFree(d_PotOut);
	cudaFree(d_PotRedOut);
	cudaFree(d_r);
	cudaFree(d_vel);

	return 0;
}