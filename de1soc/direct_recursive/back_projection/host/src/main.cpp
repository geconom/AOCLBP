///////////////////////////////////////////////////////////////////////////////////
// This OpenCL application executs the Backprojection Algorithm on an Altera FPGA.
// The kernel is defined in a device/fft1d.cl file.  The Altera 
// Offline Compiler tool ('aoc') compiles the kernel source into a 'fft1d.aocx' 
// file containing a hardware programming image for the FPGA.  The host program 
// provides the contents of the .aocx file to the clCreateProgramWithBinary OpenCL
// API for runtime programming of the FPGA.
//
// When compiling this application, ensure that the Altera SDK for OpenCL
// is properly installed.
///////////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <assert.h>
#include <cstring> 
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#define MAX_SOURCE_SIZE (0x100000)

/* Constants */
#define NUM_PARTS 4                       //right now, this program does not support change of this number
#define ERROR_TOLERANCE 0.0001            //EPSILON
#define M_PI 3.14159265358979323846
#define OUTPUT_FILENAME "outputImage1.txt" 
#define OUT1 "curr.txt"
#define LINEAR_SUPPORT 1
#define RADIAL_SUPPORT LINEAR_SUPPORT
const float sinf1[256]={1.000000,0.999925,0.999699,0.999322,0.998795,0.998118,0.997290,0.996313,0.995185,0.993907,0.992480,0.990903,0.989177,0.987301,0.985278,0.983105,0.980785,0.978317,0.975702,0.972940,0.970031,0.966976,0.963776,0.960431,0.956940,0.953306,0.949528,0.945607,0.941544,0.937339,0.932993,0.928506,0.923880,0.919114,0.914210,0.909168,0.903989,0.898674,0.893224,0.887640,0.881921,0.876070,0.870087,0.863973,0.857729,0.851355,0.844854,0.838225,0.831470,0.824589,0.817585,0.810457,0.803208,0.795837,0.788346,0.780737,0.773010,0.765167,0.757209,0.749136,0.740951,0.732654,0.724247,0.715731,0.707107,0.698376,0.689541,0.680601,0.671559,0.662416,0.653173,0.643832,0.634393,0.624859,0.615232,0.605511,0.595699,0.585798,0.575808,0.565732,0.555570,0.545325,0.534998,0.524590,0.514103,0.503538,0.492898,0.482184,0.471397,0.460539,0.449611,0.438616,0.427555,0.416429,0.405241,0.393992,0.382683,0.371317,0.359895,0.348419,0.336890,0.325310,0.313682,0.302006,0.290285,0.278520,0.266713,0.254866,0.242980,0.231058,0.219101,0.207111,0.195090,0.183040,0.170962,0.158858,0.146731,0.134581,0.122411,0.110222,0.098017,0.085797,0.073564,0.061321,0.049068,0.036807,0.024541,0.012271,-0.000000,-0.012271,-0.024541,-0.036807,-0.049068,-0.061321,-0.073565,-0.085797,-0.098017,-0.110222,-0.122411,-0.134581,-0.146730,-0.158858,-0.170962,-0.183040,-0.195090,-0.207111,-0.219101,-0.231058,-0.242980,-0.254866,-0.266713,-0.278520,-0.290285,-0.302006,-0.313682,-0.325310,-0.336890,-0.348419,-0.359895,-0.371317,-0.382683,-0.393992,-0.405241,-0.416430,-0.427555,-0.438616,-0.449611,-0.460539,-0.471397,-0.482184,-0.492898,-0.503538,-0.514103,-0.524590,-0.534998,-0.545325,-0.555570,-0.565732,-0.575808,-0.585798,-0.595699,-0.605511,-0.615232,-0.624860,-0.634393,-0.643831,-0.653173,-0.662416,-0.671559,-0.680601,-0.689541,-0.698376,-0.707107,-0.715731,-0.724247,-0.732654,-0.740951,-0.749136,-0.757209,-0.765167,-0.773010,-0.780737,-0.788346,-0.795837,-0.803208,-0.810457,-0.817585,-0.824589,-0.831469,-0.838225,-0.844853,-0.851355,-0.857729,-0.863973,-0.870087,-0.876070,-0.881921,-0.887640,-0.893224,-0.898675,-0.903989,-0.909168,-0.914210,-0.919114,-0.923880,-0.928506,-0.932993,-0.937339,-0.941544,-0.945607,-0.949528,-0.953306,-0.956940,-0.960431,-0.963776,-0.966977,-0.970031,-0.972940,-0.975702,-0.978317,-0.980785,-0.983105,-0.985278,-0.987301,-0.989177,-0.990903,-0.992480,-0.993907,-0.995185,-0.996313,-0.997290,-0.998118,-0.998795,-0.999322,-0.999699,-0.999925};
const float cosf1[256]={-0.000000,-0.012271,-0.024541,-0.036807,-0.049068,-0.061321,-0.073565,-0.085797,-0.098017,-0.110222,-0.122411,-0.134581,-0.146730,-0.158858,-0.170962,-0.183040,-0.195090,-0.207111,-0.219101,-0.231058,-0.242980,-0.254866,-0.266713,-0.278520,-0.290285,-0.302006,-0.313682,-0.325310,-0.336890,-0.348419,-0.359895,-0.371317,-0.382683,-0.393992,-0.405241,-0.416430,-0.427555,-0.438616,-0.449611,-0.460539,-0.471397,-0.482184,-0.492898,-0.503538,-0.514103,-0.524590,-0.534998,-0.545325,-0.555570,-0.565732,-0.575808,-0.585798,-0.595699,-0.605511,-0.615232,-0.624859,-0.634393,-0.643832,-0.653173,-0.662416,-0.671559,-0.680601,-0.689541,-0.698376,-0.707107,-0.715731,-0.724247,-0.732654,-0.740951,-0.749136,-0.757209,-0.765167,-0.773010,-0.780737,-0.788346,-0.795837,-0.803208,-0.810457,-0.817585,-0.824589,-0.831470,-0.838225,-0.844854,-0.851355,-0.857729,-0.863973,-0.870087,-0.876070,-0.881921,-0.887640,-0.893224,-0.898674,-0.903989,-0.909168,-0.914210,-0.919114,-0.923880,-0.928506,-0.932993,-0.937339,-0.941544,-0.945607,-0.949528,-0.953306,-0.956940,-0.960431,-0.963776,-0.966976,-0.970031,-0.972940,-0.975702,-0.978317,-0.980785,-0.983105,-0.985278,-0.987301,-0.989177,-0.990903,-0.992480,-0.993907,-0.995185,-0.996313,-0.997290,-0.998118,-0.998795,-0.999322,-0.999699,-0.999925,-1.000000,-0.999925,-0.999699,-0.999322,-0.998795,-0.998118,-0.997290,-0.996313,-0.995185,-0.993907,-0.992480,-0.990903,-0.989177,-0.987301,-0.985278,-0.983105,-0.980785,-0.978317,-0.975702,-0.972940,-0.970031,-0.966976,-0.963776,-0.960431,-0.956940,-0.953306,-0.949528,-0.945607,-0.941544,-0.937339,-0.932993,-0.928506,-0.923880,-0.919114,-0.914210,-0.909168,-0.903989,-0.898674,-0.893224,-0.887640,-0.881921,-0.876070,-0.870087,-0.863973,-0.857729,-0.851355,-0.844854,-0.838225,-0.831470,-0.824589,-0.817585,-0.810457,-0.803208,-0.795837,-0.788346,-0.780737,-0.773011,-0.765167,-0.757209,-0.749136,-0.740951,-0.732654,-0.724247,-0.715731,-0.707107,-0.698376,-0.689541,-0.680601,-0.671559,-0.662416,-0.653173,-0.643832,-0.634393,-0.624860,-0.615232,-0.605511,-0.595699,-0.585798,-0.575808,-0.565732,-0.555570,-0.545325,-0.534998,-0.524590,-0.514103,-0.503538,-0.492898,-0.482184,-0.471397,-0.460539,-0.449611,-0.438616,-0.427555,-0.416429,-0.405242,-0.393992,-0.382684,-0.371317,-0.359895,-0.348419,-0.336890,-0.325310,-0.313682,-0.302006,-0.290285,-0.278520,-0.266713,-0.254865,-0.242980,-0.231058,-0.219101,-0.207111,-0.195090,-0.183040,-0.170962,-0.158858,-0.146730,-0.134581,-0.122411,-0.110222,-0.098017,-0.085798,-0.073565,-0.061321,-0.049068,-0.036807,-0.024541,-0.012272};

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

/*Global Variables*/

float T;
float oneOverT;
int verbose;         //for debugging
float* sine;
float* cosine;
int baseSize,sumi;
int dwnsplSize;
cl_mem b1,b2,b3,bufferSum,bufferSino,bufferTau,bufferSine,bufferCos;
float* pixel=NULL;

typedef struct{

  int num;                       //number of angles
  int size;                      //length of each filtered sinograms
  float T;
  float** sino;
  float* sine;
  float* cosine;

} sinograms;

typedef struct{

  int size;
  float** pixel;                //pixel[0] refers to the lowest row of the image, pixel[i][0] refers to the 
} image;

typedef struct{
	cl_uint numPlatforms;
	cl_platform_id *platforms;
	cl_int status;
	cl_uint numDevices;
	cl_device_id *devices ;
	cl_command_queue cmdQueue;
	cl_context context;
	cl_program program;
	cl_kernel kernel;
} device_info;

void cleanup();
static void display_device_info( cl_device_id device );
device_info* dev;

float myTheta(int p,int P)
{
  return M_PI*p/P+M_PI/2;
}

// Function that initializes the devices' fields once!
void initialize_devices(device_info* dev)

{
    //Step1 :discover and initialize platforms

	dev->numPlatforms= 0;
	dev->platforms= NULL;
	dev->status = clGetPlatformIDs(0, NULL, &(dev->numPlatforms));
	dev->platforms=(cl_platform_id*)malloc((dev->numPlatforms)*sizeof(cl_platform_id));
	dev->status= clGetPlatformIDs((dev->numPlatforms), (dev->platforms),NULL);
	
	// User-visible output - Platform information
    {
      char char_buffer[STRING_BUFFER_LEN]; 
      printf("Querying platform for info:\n");
      printf("==========================\n");
      clGetPlatformInfo((dev->platforms)[0], CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
      printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
      clGetPlatformInfo((dev->platforms)[0], CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
      printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
      clGetPlatformInfo((dev->platforms)[0], CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
      printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
    }

  
	//Step2 :Discover and initialize devices

	dev->numDevices= 0;
	dev->devices = NULL;
	dev->status =clGetDeviceIDs((dev->platforms)[0],CL_DEVICE_TYPE_ALL,0,NULL,&(dev->numDevices));
	dev->devices=(cl_device_id*)malloc((dev->numDevices)*sizeof(cl_device_id));
	dev->status =clGetDeviceIDs((dev->platforms)[0],CL_DEVICE_TYPE_ALL,dev->numDevices,dev->devices,NULL);
    
	// Display some device information.
    display_device_info((dev->devices)[0]);

	//Step3 :Create a context

	dev->context=NULL;
	dev->context = clCreateContext(NULL,dev->numDevices,dev->devices,NULL,NULL,&(dev->status));

	//Step4:Create a command queue

	dev->cmdQueue=clCreateCommandQueue(dev->context,(dev->devices)[0],0,&(dev->status));

	//Step7:Create and compile the program

	dev->program = createProgramFromBinary(dev->context,"kernels.aocx",&((dev->devices)[0]),1);
	(dev->status )=clBuildProgram(dev->program,dev->numDevices,dev->devices,NULL,NULL,NULL);

	return ;
}

/*
 * takes care of freeing pointers inside sinograms structure
 *
 */
void freeSino(sinograms* sino)
{
  int num = sino->num;
  int i;
  float** curr=sino->sino;

  for(i=0;i<num;i++){
    free(curr[i]);
  }

  free(sino->sine);
  free(sino->cosine);
}


/*
 * caller free both sino and newSino
 * 
 * This function prepares a new set of sinograms for next recursive call w/o downsampling the sinograms angularly.
 *
 */
sinograms* newSinoForNextIter2(sinograms* sino,int newSinoSize,float constShiftingFactor,float* nu_i)
{

  int P = sino->num;
  sinograms* newSino;
  int size = sino->size;
  int newNumSino;

  int m,n,k,p;
  float kk,pp;
  float sum;
  float angle;

  
  float temp1,temp2,temp3;
  int shiftingFactor;

  float radialShift;
  float totalShift;
  int n2;
  int lowerPBound,upperPBound,lowerKBound,upperKBound;

  float radialSupport;

  //preparing new sinograms
  newSino = (sinograms*)malloc(1*sizeof(sinograms));
  newSino->T = T;

  newSino->size = newSinoSize; 
  newSino->num = P;

  newNumSino = newSino->num;
  (newSino->sino)  = (float**)malloc(newNumSino*sizeof(float*));
  (newSino->sine) = (float*)malloc(newNumSino*sizeof(float));
  (newSino->cosine) = (float*)malloc(newNumSino*sizeof(float));

  radialSupport = RADIAL_SUPPORT;

  //for each angle of the new set of sinograms
  for(n=0;n<newNumSino;n++){
    //copy from old sinograms directly
    (newSino->sino)[n] = (float*)malloc(newSinoSize*sizeof(float));
    (newSino->sine)[n] = (sino->sine)[n];
    (newSino->cosine)[n] = (sino->cosine)[n];    
    
    shiftingFactor = rintf(nu_i[n]) + constShiftingFactor;      

    //for each entry in each of the new sinograms
    for(m=0;m<newSinoSize;m++){
      //we just copy the values from old ones, because we are not downsampling or filtering here.
      (newSino->sino)[n][m] = (sino->sino)[n][m-shiftingFactor];
	  
    }//end for m
  }//end for n
  
  return newSino;
}
//--------------------------------------------------------------


/*
 * computes 'nu' for given sine/cosine values and given shifting constants (\delta)
 *
 */
float nu(float tau,float shiftX,float shiftY,float sine,float cosine)//,float oneOverT)
{
  float temp1;
  float temp2;
  
  temp1 = (shiftX)*cosine;
  temp2 = (shiftY)*sine;

  return tau - (temp1+temp2);
}


/*
 * This function is the top-level function for recursive backprojection.
 * It makes a call to 'direct' once size<=baseSize.
 */

/*
 * direct(the sinograms,the size of the image, tau's,output image)
 *
 * direct corresponds EXACTLY to the direct method. 
 *
 */


double direct(device_info* devi,sinograms* g,int size,float* tau,image* ans)
{
  int m;        //x direction
  int n;        //y direction
  float temp;
  float* Sino=NULL;
  float* TSine=NULL;
  float* TCosine=NULL;
  float* pixel=NULL;
  float* curRow; 
  float middle = ((float)(g->size))/2-0.5;
  float halfSize = 0.5*size;
  int numSino = g->num;
  int sinsize = g->size;
  float scale = M_PI/numSino;      

  FILE* output1;
  char* f1=OUT1;

  pixel=(float*)malloc(size*size*sizeof(float));
  Sino=(float*)malloc(sinsize*numSino*sizeof(float));
  TSine=(float*)malloc(numSino*sizeof(float));
  TCosine=(float*)malloc(numSino*sizeof(float));
  
  for(n=0;n<numSino;n++){
	TSine[n]=(g->sine)[n];
	TCosine[n]=(g->cosine)[n];
	for(m=0;m<sinsize;m++){
		Sino[n*sinsize+m]=(g->sino)[n][m];
	}
  }

  //printf("test1\n");      	
  int elements =numSino;
  int elements1=size*size;
  int elements2=size;
  size_t datasize =elements;
  size_t datasize1=elements*sinsize;
  size_t datasize2=elements1;


  //Step5:Create device buffers	
  bufferSino = clCreateBuffer(devi->context,CL_MEM_READ_ONLY,datasize1*sizeof(float),NULL,&(devi->status));
  bufferSum = clCreateBuffer(devi->context,CL_MEM_WRITE_ONLY,datasize2*sizeof(float),NULL,&(devi->status));
  bufferTau = clCreateBuffer(devi->context,CL_MEM_READ_ONLY,datasize*sizeof(float),NULL,&(devi->status));
  bufferSine = clCreateBuffer(devi->context,CL_MEM_READ_ONLY,datasize*sizeof(float),NULL,&(devi->status));
  bufferCos = clCreateBuffer(devi->context,CL_MEM_READ_ONLY,datasize*sizeof(float),NULL,&(devi->status));

	
  //Step6:Write host and data to device buffers
  (devi->status)=clEnqueueWriteBuffer(devi->cmdQueue,bufferSino,CL_FALSE,0,datasize1*sizeof(float),Sino,0,NULL,NULL);
  (devi->status)=clEnqueueWriteBuffer(devi->cmdQueue,bufferTau,CL_FALSE,0,datasize*sizeof(float),tau,0,NULL,NULL);
  (devi->status)=clEnqueueWriteBuffer(devi->cmdQueue,bufferSine,CL_FALSE,0,datasize*sizeof(float),TSine,0,NULL,NULL);
  (devi->status)=clEnqueueWriteBuffer(devi->cmdQueue,bufferCos,CL_FALSE,0,datasize*sizeof(float),TCosine,0,NULL,NULL);	
	
  //Step8:Create the kernel
  devi->kernel=NULL;
  devi->kernel=clCreateKernel(devi->program,"sum_sino",&(devi->status));
	
  //Step9:Set the rest of kernel arguments
	
  (devi->status)=clSetKernelArg(devi->kernel,0,sizeof(cl_mem),&bufferSino);
  (devi->status)=clSetKernelArg(devi->kernel,1,sizeof(cl_mem),&bufferSum);
  (devi->status)=clSetKernelArg(devi->kernel,2,sizeof(float),&middle);
  (devi->status)=clSetKernelArg(devi->kernel,3,sizeof(float),&halfSize);
  (devi->status)=clSetKernelArg(devi->kernel,4,sizeof(int),&numSino);
  (devi->status)=clSetKernelArg(devi->kernel,5,sizeof(int),&size);
  (devi->status)=clSetKernelArg(devi->kernel,6,sizeof(cl_mem),&bufferTau);
  (devi->status)=clSetKernelArg(devi->kernel,7,sizeof(cl_mem),&bufferSine);
  (devi->status)=clSetKernelArg(devi->kernel,8,sizeof(cl_mem),&bufferCos);
  (devi->status)=clSetKernelArg(devi->kernel,9,sizeof(int),&sinsize);
  (devi->status)=clSetKernelArg(devi->kernel,10,sizeof(float),&scale);
  (devi->status)=clSetKernelArg(devi->kernel,11,sizeof(float),&oneOverT);
 
  printf("\n\nKernel initialization is complete.\n");

  // Get the iterationstamp to evaluate performance
  double time = getCurrentTimestamp();

  //Step10:Configure the work-item structure

  size_t globalWorkSize[2]={size,size};
	
  //Step11:Enqueue the kernel for execution
  (devi->status)= clEnqueueNDRangeKernel(devi->cmdQueue,devi->kernel,2,NULL,globalWorkSize,NULL,0,NULL,NULL);
  (devi->status)= clFinish(devi->cmdQueue);

  // Record execution time
  time = getCurrentTimestamp() - time;

  //Step12:Read the output buffer back to the host
	
  clEnqueueReadBuffer(devi->cmdQueue,bufferSum,CL_TRUE,0,datasize2*sizeof(float),pixel,0,NULL,NULL);
  
  //printf("\tKernel processing time = %.4fms\n", (float)(time * 1E3));
  //printf("test2");
  for(n=0;n<size;n++){
    for(m=0;m<size;m++){
	(ans->pixel)[n][m]=pixel[n*size+m];
    } 
  } 
    
  //Step13:Release OpenCl resources

  clReleaseMemObject(bufferSum);
  clReleaseMemObject(bufferSino);
  clReleaseMemObject(bufferTau);
  clReleaseMemObject(bufferSine);
  clReleaseMemObject(bufferCos);

  free(Sino);
  free(TSine);
  free(TCosine);
  free(pixel);
  pixel=NULL;
  return time;
}


double bp(device_info* devic,sinograms* sino,int size,float* tau,image* ans,double time_in)
{
  sinograms* newSino;
  image* subImage;

  int i,j,m,n,k,p,u,v;

  float* curRow;
  float temp1,temp2;

  int pp;

  int newSize;
  int numSino;
  int sinoSize;
  int offsetX;
  int offsetY;

  float shiftX,shiftY; 

  float* nu_i;
  float* nu_temp;
  float nu_p,nu_2n;

  int P = sino->num;
  float angle;

  float sum=0;
  float shift;

  int newSinoSize;
  int newNumSino;
  int constShiftingFactor;
  double time_out;
  double time[NUM_PARTS+1];

  if(size<=baseSize || sino->num <=1){ 
    //BASE CASE!!!!

    time_out=direct(devic,sino,size,tau,ans);  

    return time_out;

  } else {
    //GENERAL CASE
  
    subImage = (image*)malloc(1*sizeof(image));          //prepare subImage for recursive call
    newSize = size/2;
    numSino = (sino->num);
    sinoSize = (sino->size);

    subImage->size = newSize;
    (subImage->pixel) = (float**)malloc(newSize*sizeof(float*));
    shift = (float)size/4 * oneOverT;  //save some computations when computing nu's by premultiply with 1/T

    newSinoSize = ceilf((float)sinoSize/2) + (sinoSize%4==3 || sinoSize%4==2?1:0);
    constShiftingFactor = ((float)newSinoSize/2) - ((float)sinoSize/2);    

    //for each part (that we break the image into)
	time[0]=time_in;
    for(i=0;i<NUM_PARTS;i++){

      if(i==0){
	shiftX = shift;
	shiftY = shift;
      } else if(i==1){
	shiftX = -shift;
	shiftY = shift;
      } else if(i==2){
	shiftX = -shift;
	shiftY = -shift;
      } else {
	shiftX = shift;
	shiftY = -shift;
      }


      /************/

      nu_i = (float*)malloc(numSino*sizeof(float));
      nu_temp = (float*)malloc(numSino*sizeof(float));

      for(p=0;p<numSino;p++){

	nu_i[p] = nu(tau[p],shiftX,shiftY,(sino->sine)[p],(sino->cosine)[p]);//,oneOverT);
	nu_temp[p] = rintf(nu_i[p]);
      }

      //nu_i are real nu's (for this quadrant, for all theta's)
      //nu_temp are rounded nu's

	newSino = newSinoForNextIter2(sino,newSinoSize,constShiftingFactor,nu_i);
	pp = 1;   
    

      /************/


      
      //manipulate the right part of the image
      if(i<2){
	// 0 and 1
	offsetY = newSize;

      } else {
	// 2 and 3
	offsetY = 0;
      }

      if(i==0 || i==3){
	// 0 and 3
	offsetX = newSize;
      } else {
	// 1 and 2
	offsetX = 0;
      }

      //make subImages point to appropriate parts of the big image

      for(m=0;m<newSize;m++){
	(subImage->pixel)[m] = (ans->pixel)[m+offsetY] + offsetX;
      }
      
      newNumSino = (newSino->num);

      //for each nu_p, compute <nu_p>

      for(p=0;p<newNumSino;p++){
	nu_i[p] = nu_i[pp*p]-nu_temp[pp*p];
      }

      //call bp recursively
      time[i+1]=bp(devic,newSino,newSize,nu_i,subImage,time[i])+time[i];    
      

      freeSino(newSino);
      free(newSino);
      newSino = NULL;
      free(nu_i);
      free(nu_temp);
      nu_i = NULL;
    
    }//end for i

	time_out=time[NUM_PARTS];
    free(subImage);
    free(subImage->pixel);
    //do not free subImage's pixels, because they are the output!
    
    return time_out;

  }//end if not base case


}//end function bp


/*
 * function to read in input sinograms from file.
 * 
 * The format of the input file is a bit confusing.
 * Let me know if you want to know about it.
 * 
 * 
 */
sinograms* sinoFromFile(char* filename,float* s1, float* c1)
{
  FILE* input;
  //FILE* trigInput;
  //char* trigFilename = TRIG_FILENAME;
  sinograms* sino;
  int numSino;
  int i=0;
  int sinoSize;
  char* curValue = (char*)malloc(25*sizeof(char));
  int j=0;
  float T;
  char* exponent;
  int exp;
  char* Ttemp = (char*)malloc(5*sizeof(char));


  sino = (sinograms*)calloc(1,sizeof(sinograms));

  if((input = fopen(filename,"r"))==0){
    printf("can't open file %s\n",filename);
    exit(0);
  }
  
  
  
  fscanf(input,"%d",&numSino);
  fscanf(input,"%d",&sinoSize);
  fscanf(input,"%s",Ttemp);
  T = atof(Ttemp);


  sino->num = numSino;
  sino->size = sinoSize;
  sino->T = T;
  sino->sino = (float**)calloc(numSino,sizeof(float*));
  sino->sine = (float*)calloc(numSino,sizeof(float));
  sino->cosine = (float*)calloc(numSino,sizeof(float));

  
 
  //loading trig values
  for(i=0;i<numSino;i++){
   // fscanf(trigInput,"%s",curValue);
    (sino->sine)[i] = s1[i];
    //fscanf(trigInput,"%s",curValue);
    (sino->cosine)[i] = c1[i];

    
    (sino->sino)[i] = (float*)calloc(sinoSize,sizeof(float));
  }//end for i

  //reading in actual sinogram data
  for(j=0;j<sinoSize;j++){
    
    for(i=0;i<numSino;i++){
      
      fscanf(input,"%s",curValue);
      if((exponent = strchr(curValue,'e'))){

	   exponent[0] = 0;  //terminate mantissa
	   exponent = exponent + 1;  //skip 'e'

	   exp = atoi(exponent);
	   (sino->sino)[i][j] = atof(curValue)*powf(10,exp);

      }else {

	   (sino->sino)[i][j] = atof(curValue);
      }
      
    }//end for i
    
  }//end for j

  
  fclose(input);
  free(curValue);
  free(Ttemp);
  
  return sino;
}



/*
 * This function precomputes sin's and cos's of P angles--uniformly spaced between Pi/2 and 3Pi/2.
 * Results are stored in the file specified by TRIG_FILENAME.
 *
 */
void precomputeTrig(int P,float* s1, float* c1)
{
  int i;
  
  for(i=0;i<P;i++){
    s1[i]=sinf1[i];
    c1[i]=cosf1[i];
	
  }

}
 
int main(int argc,char* argv[])
{


  char* filename = (char*)malloc(20*sizeof(char*));
  sinograms* sino;
  int i;
  int j;
  int numSino,sinoSize;
  int size;
  image* img = (image*)malloc(1*sizeof(image)); 
  float* tau;
  float* curRow;

  FILE* output;
  char* filename2 = OUTPUT_FILENAME;
  int tempNumSino;

    /*Step1 Read Input*/
    
	// argument1 is the number of sinograms
	if(argv[1]==NULL){
	printf("Number of Sinograms not set\n");
		exit(0);
	}
	numSino = atoi(argv[1]);
	
	//argument2 is the image size (output)
	if(argv[2]==NULL){
		printf("Image size not set\n");
		exit(0);
	}
	size = atoi(argv[2]);
 
  //argument3 is the base case size B
	if(argv[3] == NULL){
		printf("baseSize not set\n");
		exit(0);
	}

	baseSize = atoi(argv[3]);
  
	//argument4 is the input filename (containing sinograms)
	if(argv[4]==NULL){
		printf("filename not set\n");
		exit(0);
	}
	strcpy(filename,argv[4]);

    /*Step2 Precompute Trig and Initialize Devices*/
    
	sine = (float*)malloc(numSino*sizeof(float));
	cosine = (float*)malloc(numSino*sizeof(float));
	
	dev = (device_info*)calloc(1,sizeof(device_info));
	initialize_devices (dev);
	precomputeTrig(numSino,sine,cosine);

								
    /*Step3 Read Sinograms from File*/
															
	sino = sinoFromFile(filename,sine,cosine);
	
	numSino = sino->num;	
	sinoSize=sino->size;
    	
		 
 	(img->size) = size;
  	(img->pixel) = (float**)calloc(size,sizeof(float*));
  

  	for(i=0;i<size;i++){
    		(img->pixel)[i] = (float*)calloc(size,sizeof(float));
  	}
  	
	tau = (float*)calloc(sino->num,sizeof(float));
  	T = sino->T;
  	oneOverT = 1/T;

    /*Step4 Execute the direct method */	
	// Get the iterationstamp to evaluate performance
    double time = bp(dev,sino,size,tau,img,0);

    printf("\tOverall processing time = %.4fms\n", (float)(time * 1E3));

    /*Step5 output image values to file (for used in matlab)*/

  	if(!(output = fopen(filename2,"w"))){
    		printf("Could not open file output.txt\n");
    		exit(0);
  	}
  

  	for(i=size-1;i>=0;i--){
    	   for(j=0;j<size;j++){
      		fprintf(output,"%f ",pixel[i*size+j]);  
    	   }
    		fprintf(output,"\n");
    	}

  	fclose(output);

    /*Step6 Free arrays and structs*/
 
  	freeSino(sino);
  	free(sino);
	free(pixel);
  	free(filename);
 
  	return 0;
}

// Free the resources allocated during initialization in the fft1d example. Here, dummy, only for correct linking
void cleanup() {
/*  if(kernel) 
    clReleaseKernel(kernel);  
  if(program) 
    clReleaseProgram(program);
  if(queue) 
    clReleaseCommandQueue(queue);
  if(context) 
    clReleaseContext(context);
  if(h_inData)
	alignedFree(h_inData);
  if (h_outData)
	alignedFree(h_outData);
  if (h_verify)
	alignedFree(h_verify);
  if (d_inData)
	clReleaseMemObject(d_inData);
  if (d_outData) 
	clReleaseMemObject(d_outData); */
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %lu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info( cl_device_id device ) {

   printf("Querying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}
