//#include <math.h>

float weight(float exactRadialPosition,int k)
{
  #define RADIAL_INTERP(x) LINEAR_INTERP(x)
  #ifndef LINEAR_INTERP
  #define LINEAR_INTERP(x) (1.0-fabs(x))
  #endif
  
  float argToIntp;
  float ans;

  argToIntp = exactRadialPosition-k;     //the unit of argToIntp is in samples
  ans = RADIAL_INTERP(argToIntp);
  
  return ans;
}


__attribute__((num_compute_units(2)))
__kernel void sum_sino(__global float* restrict bufsino,__global float* restrict bufsum,float mid,float halfs,int nsino,int siz,__global float* restrict buftau,__global float* restrict bufsine,__global float* restrict bufcos,int sinoSize,float scal,float ooT)
  {

  #define ERROR_TOLERANCE 0.0001
  #define RADIAL_SUPPORT LINEAR_SUPPORT
  #define LINEAR_SUPPORT 1

  int m=get_global_id(0);
  int n=get_global_id(1);
  int ip=n*siz+m;
  int p;
  int k;
  int P=nsino;
  float i;
  float my_sum=0;
  float temp;

  float realm=m-halfs+0.5;      
  float realn=n-halfs+0.5;   

  float scaledRealm = realm*ooT;
  float scaledRealn = realn*ooT;

  float lowerXLimit;
  float upperXLimit;
  float lowerYLimit;
  float upperYLimit;

  float intpSupport;

  float curr;
  float lowerKBound;
  float upperKBound;

  int lowerKBound2;
  int upperKBound2;

  float theta;

  float exactRadialPosition;
  int lower_tolerance;
  int upper_tolerance;

  intpSupport = RADIAL_SUPPORT;

  for(p=0;p<P;p++){

     exactRadialPosition = scaledRealm*bufcos[p] + scaledRealn*bufsine[p];
     exactRadialPosition -= buftau[p]-mid; 

     lowerKBound = exactRadialPosition-intpSupport;
     upperKBound = exactRadialPosition+intpSupport;

     lower_tolerance = (int) (lowerKBound - ERROR_TOLERANCE + 1);
     upper_tolerance = (int) upperKBound;

     lowerKBound2 = (lower_tolerance > 0 ? lower_tolerance : 0);
     upperKBound2 = (upper_tolerance < (sinoSize-1) ? upper_tolerance : (sinoSize-1));   

    if(upperKBound2<lowerKBound2){
      upperKBound2 = lowerKBound2;
  
     }

    for(k=lowerKBound2;k<=upperKBound2;k++){

      curr = bufsino[p*sinoSize+k];      
      temp = weight(exactRadialPosition,k);
      my_sum += curr*temp;

    }
  } 
    
  bufsum[ip]=my_sum*scal; 
 
}



