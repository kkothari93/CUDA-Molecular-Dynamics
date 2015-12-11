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

#define N_ATOMS 64
#define MASS_ATOM 1.0
#define time_step 0.01
#define L 4.0f
#define T 7.28
#define NUM_STEPS 1000

const int BLOCK_SIZE = 1024;

//__global__ void makePairs(float* positions, float* pairpos){
//	__shared__ float pos[N_ATOMS * 3];
//	float delx;
//	float dely;
//	float delz;
//	int index = threadIdx.x + blockDim.x*blockIdx.x;
//	if (index < N_ATOMS * 3)
//		pos[index] = positions[index];
//	__syncthreads();
//	if (index < N_ATOMS){
//#pragma unroll
//		for (int i = 0; i < N_ATOMS; i++){
//			delx = pos[i] - pos[index];
//			dely = pos[N_ATOMS + i] - pos[N_ATOMS + index];
//			delz = pos[2 * N_ATOMS + i] - pos[2 * N_ATOMS + index];
//			delx -= int(delx * 2.0 / L)*L / 2.0;
//			dely -= int(dely * 2.0 / L)*L / 2.0;
//			delz -= int(delz * 2.0 / L)*L / 2.0;
//			pairpos[i + index*N_ATOMS] = delx;
//			pairpos[i + index*N_ATOMS + N_ATOMS*N_ATOMS] = dely;
//			pairpos[i + index*N_ATOMS + 2 * N_ATOMS*N_ATOMS] = delz;
//		}
//	}
//}

__global__ void makePairs(float* positions, float* pairpos){
	__shared__ float pos[N_ATOMS * 3];
	float del;
	int iatom1, iatom2;
	int tx = threadIdx.x;
	int index = tx + blockDim.x*blockIdx.x;
	if (tx < N_ATOMS * 3) { pos[tx] = positions[tx]; }
	__syncthreads();
	if (index < N_ATOMS*N_ATOMS){
		iatom1 = int(index / N_ATOMS);
		iatom2 = index - iatom1*N_ATOMS;
//#pragma unroll
		for (int i = 0; i < 3; i++){
			del = pos[iatom1 + i*N_ATOMS] - pos[iatom2 + i*N_ATOMS];
			if (fabs(del) > L / 2.0) { del += (2*(del<0)-1)*L; }
			pairpos[index + i*N_ATOMS*N_ATOMS] = del;
		}
	}
}

__global__ void init_r(float* r, int N_cube){
	int ix = threadIdx.x + blockDim.x* blockIdx.x;
	int iy = threadIdx.y + blockDim.y* blockIdx.y;
	int iz = threadIdx.z + blockDim.z* blockIdx.z;
	int index = ix + iy*N_cube + iz * N_cube * N_cube;
	r[index] = L / 2.0 * (1.0 - float(2 * ix + 1) / N_cube);
	r[index + N_ATOMS] = L / 2.0 * (1.0 - float(2 * iy + 1) / N_cube);
	r[index + 2 * N_ATOMS] = L / 2.0 * (1.0 - float(2 * iz + 1) / N_cube);
}

__device__ void PutInBox(float r){
	if (fabs(r)>L / 2.0)
		r += signbit(r) *ceil(r / L)*L;
}

__global__ void kinematics(float* positions, float* force, float* vel, int len){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int index = bx*blockDim.x + tx;
	if (index < len){
		positions[index] += 0.5 * force[index] / MASS_ATOM * time_step*time_step + vel[index] * time_step;
		PutInBox(positions[index]);
		vel[index] += force[index] / MASS_ATOM * time_step;
	}
}

/*
TTD
1. Calculate N - done
2. Divide the pairwise position array into blocks of max 1024 threads - done
3. Create an array of ending rows - done
4. Now iterate over a block and store sum in the shared memory - done
5. Create another shared memory to calculate and store the force - done
6. Reduction over the entire array to get the net force at each particle
7. Kinematics over particles to get new distances
*/

//--use_fast_math option with nvcc while compiling

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
		if (delr2 < 1.0e-8) {
			Potential = 0;
			Forcex = 0;
			Forcey = 0;
			Forcez = 0;
		}
		else{
			Potential = 4 * __fadd_rn(delrm6*delrm6, -1 * delrm6);
			Forcex = (delx / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
			Forcey = (dely / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
			Forcez = (delz / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
		}
		PotOut[row] = Potential;
		ForceOut[row] = Forcex;
		ForceOut[row + N*N] = Forcey;
		ForceOut[row + N*N * 2] = Forcez;
	}

}

__global__ void forcered_simple(float * force, float * forcered){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int i = 0;
	int findex;
	//if (index == 0){ printf("In force reduction kernel! \n"); }
	if (index < 3 * N_ATOMS){
		findex = (index / N_ATOMS)*N_ATOMS*N_ATOMS + index % 2;
		forcered[index] = 0.0;
		for (i = 0; i < N_ATOMS; i++){
			__syncthreads();
			forcered[index] += force[findex + i*N_ATOMS];
		}
	}
	/*if (index == 0){
	printf("forcered [0]= %f \n", forcered[0]);
	printf("forcered [2]= %f \n", forcered[2]);
	printf("forcered [4]= %f \n \n", forcered[4]);
	}*/
}

void init_vel(float* vel){
	srand(1.0);
	float netP[3] = { 0.0f, 0.0f, 0.0f };
	float netE = 0.0;
	int i;
	float a, b, c;
	for (i = 0; i < N_ATOMS; i++){
		a = rand() - 0.5;
		b = rand() - 0.5;
		c = rand() - 0.5;
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



int main(){

	//printf("True is given as %i\n", (4>3)*2-1);
	//printf("False is given as %i\n", (4<3) * 2 - 1);
	////printf("fabs(-2.534) is given as %f\n", fabs(-2.534));

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
	/*cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);*/
	/*for (int ii = 0; ii < N_ATOMS * 3; ii++){
		printf("%f \t", r[ii]);
		if ((ii + 1) % N_ATOMS == 0)
			printf("\n");
	}
	printf("\n All r printed \n");
	printf("Generation of r succeeded! \n");*/
	//char str[100];
	//printf("\n Enter the filename :");
	//gets(str);
	//create_csv(str, r , 3, N_ATOMS);

	// Initialise velocity
	init_vel(vel);
	// Check velocity
	//for (int ii = 0; ii < N_ATOMS*3; ii++){
	//printf("%f \t", vel[ii]);
	//if ((ii+1) % N_ATOMS == 0)
	//printf("\n");
	//}
	//printf("\n All vel printed \n");

	// Copy velocity data to GPU
	cudaMemcpy(d_vel, vel, size, cudaMemcpyHostToDevice);
	state = cudaDeviceSynchronize();
	if (state != cudaSuccess){
		printf("Init_r did not succeed: %s \n", cudaGetErrorString(state));
	}

	for (int t = 0; t<NUM_STEPS; t++){
		//////////////////////////////////////////////////////////////////////
		//! Timing CUDA code
		//////////////////////////////////////////////////////////////////////
		//cudaEvent_t start, stop;
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);

		//cudaEventRecord(start);

		//cudaEventRecord(stop);

		//cudaEventSynchronize(stop);
		//float milliseconds = 0;
		//cudaEventElapsedTime(&milliseconds, start, stop);

		//////////////////////////////////////////////////////////////////////
		makePairs << <ceil(N_ATOMS*N_ATOMS / float(BLOCK_SIZE)), BLOCK_SIZE >> >(d_r, d_pairpos);
		state = cudaDeviceSynchronize();
		if (state != cudaSuccess){
			printf("make pairs did not succeed: %s \n", cudaGetErrorString(state));
		}

		//// print pair distances
		//cudaMemcpy(pairpos, d_pairpos, size*N_ATOMS, cudaMemcpyDeviceToHost);
		//for (int i = 0; i < 3*N_ATOMS*N_ATOMS; i++){
		//	rpairpos[i] = pairpos[i];
		//	//rpairpos[i] = sqrtf(pairpos[i] * pairpos[i]
		//	//	+ pairpos[i + N_ATOMS*N_ATOMS] * pairpos[i + N_ATOMS*N_ATOMS]
		//	//	+ pairpos[i + 2 * N_ATOMS*N_ATOMS] * pairpos[i + 2 * N_ATOMS*N_ATOMS]);
		//	printf("%f \t", rpairpos[i]);
		//	if ((i + 1) % N_ATOMS == 0) printf("\n");
		//}
		//printf("\n All pair r printed \n");
		//char str1[100];
		//printf("\n Enter the filename :");
		//gets(str1);
		//create_csv(str1, rpairpos, N_ATOMS*3, N_ATOMS);
		

		int gridDim = ceil((N_ATOMS*N_ATOMS) / (float)BLOCK_SIZE);
		potForce << <gridDim, BLOCK_SIZE >> >(d_pairpos, N_ATOMS, d_PotOut, d_ForceOut);
		cudaDeviceSynchronize();
		// Check potential and force
		//cudaMemcpy(PotOut, d_PotOut, size*N_ATOMS / 3, cudaMemcpyDeviceToHost);
		//cudaMemcpy(ForceOut, d_ForceOut, size*N_ATOMS, cudaMemcpyDeviceToHost);
		//printf("Now printing non-reduced force array \n");
		//for (int ii = 0; ii < N_ATOMS*N_ATOMS * 3; ii++){
		//	printf("%1.10f \t", ForceOut[ii]);
		//	if ((ii + 1) % N_ATOMS == 0)
		//		printf("\n");
		//}


		//cudaMemcpy(d_ForceOutRed, ForceOutRed, size*N_ATOMS, cudaMemcpyHostToDevice);

		total << < gridDim, BLOCK_SIZE >> >(d_PotOut, d_PotRedOut, N_ATOMS*N_ATOMS);
		cudaDeviceSynchronize();
		int len = int(ceil(N_ATOMS*N_ATOMS / float(BLOCK_SIZE)));
		state = cudaMemcpy(PotRedOut, d_PotRedOut, len*sizeof(float), cudaMemcpyDeviceToHost);
		if (state != cudaSuccess)
			printf("Output memcpy failed %s \n", cudaGetErrorString(state));
		for (int i = 1; i < len; i++){
			PotRedOut[0] += PotRedOut[i];
		}

		if ((t + 1) % 100 == 0){ printf("Potential energy at iteration [%i] is %f \n", t, PotRedOut[0] / 2.0); }
		forcered_simple << <ceil(3 * N_ATOMS / float(BLOCK_SIZE)), BLOCK_SIZE >> >(d_ForceOut, d_ForceOutRed);
		state = cudaDeviceSynchronize();
		if (state != cudaSuccess)
			printf("Force kernel failed: %s \n", cudaGetErrorString(state));

		//check force reduction
		state = cudaMemcpy(ForceOutRed, d_ForceOutRed, 3*N_ATOMS*sizeof(float), cudaMemcpyDeviceToHost);
		if (state != cudaSuccess)
			printf("Force memcpy failed: %s \n", cudaGetErrorString(state));
		//printf("Now printing reduced force array \n");
		//for (int ii = 0; ii < N_ATOMS * 3; ii++){
		//	printf("%f \t", ForceOutRed[ii]);
		//	if ((ii + 1) % N_ATOMS == 0)
		//		printf("\n");
		//}
		kinematics << < ceil((N_ATOMS * 3) / (float)BLOCK_SIZE), BLOCK_SIZE >> >(d_r, d_ForceOutRed, d_vel, 3 * N_ATOMS);

		cudaDeviceSynchronize();
		//cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);
		//for (int ii = 0; ii < N_ATOMS * 3; ii++){
		//	printf("%1.10f \t", r[ii]);
		//	if ((ii + 1) % N_ATOMS == 0)
		//		printf("\n");
		//}
		//printf("\n All new r printed \n");
	} // } for time integration for loop*/


	// Free up host memory
	free(r);
	free(vel);
	free(pairpos);
	free(rpairpos);

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

//for i = 1:steps
// init_r;
// init_vel;
// makePairs kernel
// potForce_1 kernel
// potForce_2 kernel
// kinematics kernel
// PutInBox kernel
// end