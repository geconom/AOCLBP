/*# Copyright (c) 2004 Carnegie Mellon University
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# A copy of the GNU General Public License can be obtained from 
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330, 
# Boston, MA  02111-1307  USA */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
 * This program implements the hierachical fast backprojection algorithm.
 * 
 * Author: Thammanit Pipatsrisawat (Knot)
 * Email addresses: thammaknot@cmu.edu, thammaknot@hotmail.com, thammaknot@gmail.com
 * Start Date: 6/13/04
 * Last modified: 10/01/04
 *
 * Please read 'Conventions' section right below 'Modification Log' section.
 * It might help clarify things.
 *
 * This work was done as part of the SPIRAL project: www.spiral.net
 */


/*** COMPILATION AND USAGE ***
 *
 * This program consists of only 1 file (bp_final.c). The following
 * compilation options might be considered.
 *
 * gcc: 
 * > gcc -lm -O2  bp_final.c
 *
 * icc:
 * > icc -fast bp_final.c
 *
 * Please note that, depending on machine specifications, different
 * compiler optimization flags might give different results.
 *
 * *************************************************************************
 * Before each reconstruction, you must precompute the sine and cosine
 * values for the interested number of angles.  This can be done by
 * calling (assuming a.out is the output file from compilation):
 *
 * > ./a.out p <# of angles>
 *
 * Once that is done, a real reconstruction can be perform with the
 * following arguments:
 *
 * > ./a.out <size of target image> <verbose flag> <input filename> <B> <D>
 *
 * Descriptions:
 *     - <size of target image> is the one-dimensional size of the reconstructed image. 
 *          This number must be a power of two. 
 *          The reconstructed image is assumed to be a square image.
 *     - <verbose flag> is a flag for printing some internal states during execution. 
 *          This is for debugging purpose only. Non-zero flag means no printing.
 *          If you want to add more printing statements in the code, you could consider 
 *          using the following form:
 *                                                
 *                          if(verbose==<number>){       // or (verbose< <number>)
 *                             printf("Hello World!");
 *                          }
 *                      
 *          With this scheme, we can turn on/off different levels of debugging statements.
 *
 *     - <input filename> is the name of the input file that contains filtered sinograms data.
 *     - <B> is the base case size, at which the program stops recursion.
 *     - <D> is the size of the image at which the program starts angularly downsampling the 
 *          filtered sinograms.
 *
 * for example, if we want to reconstruct an image of size 512x512
 * using 1024 projections and the filtered sinograms are stored in
 * input.txt, we do the following.
 *
 * > ./a.out p 1024    <return>
 * > ./a.out 512 0 input.txt 128 256    <return>
 *
 * This will reconstruct the image using B=128,D=256, without any debugging messages.
 * 
 *** END COMPILATION AND USAGE ***
 */



/* Modification Log 
 *
 * optRec1 : Recursive uses separate functions to prepare sinograms
 * for next recursive call.  Some inefficiencies were moved out from
 * for loops in shiftSino,filterSino,truncateSino.
 *
 * Modifications: Some inefficient computations were moved out of
 * loops in weight and sumSino.
 *
 * New in optRec2: Some inefficent computations were moved out of loops in newSinoForNextIter.
 *               : precompute shift=(float)size/4 in loop in bp (general case)
 * 
 * New in optRec3: Integration is removed. Delta function sampling function is assumed. !
 *
 * New in optRec4: The function newSinoForNextIter is improved so that it only goes through 
 *                   certain entries in the old sinograms. (bounds used) 
 *
 * New in optRec5: Divisions are converted to multiplications. It turns out that, at the moment,
 *                   changing from /2 to *0.5 slows down the performance.
 *                 : IMPORTANT- the argument to interpolating functions is now the number of entries
 *                   IN SINOGRAM MEASURE UNIT away from the point of interest.
 *                   Earlier it was the distance in PIXEL UNIT.
 *                 : Also add ability to do polynomial interpolations.
 *                 : Also add ability to do natural cubic spline (3 data points).
 *
 * New in optRec6: Precompute sin's, cos's.
 *                  : Try new angularInterp (still experimenting)--put on hold
 *
 * New in opt7: Optimize direct method more
 *               - take (m*cos+n*sine)/T out of function weight
 *               - simplify computations for lower/upperKBound, assuming delta function sampling
 *               - loop over k (int) instead of i (float) (inside sumSino)
 *               - precompute halfSize/middle in bp1 (instead of inside sumSino)
 *               - change rounding method in sumSino (ceil/floor -> rint)
 *
 * New in opt8: Optimize recurisve method more
 *               - *0.5 -> /2
 *               - precompute constShiftinFactor (newSinoForNextIter)
 *               - premultiply oneOverT to shift (bp)
 *               - do not recompute mu's when go into newSinoForNextIter
 *               - get rid of pp (float) [use p(int) instead] (newSinoForNextIter)
 *               - do not remalloc nu_i  (bp)
 *               - cache newSino->num  (bp)
 *               - Linear interpolator does not check if x is out of bound  (linear interp)
 *               - use fabs instead of if else (linear interp)
 *
 * New in opt9: Change from float -> float (which can,then, be defined as either float or double)
 *              Trying to come up with better angular downsampler and a better way to decimate projections
 *              Parameterized some features (baseSize)
 *
 * New in opt10: Define macros for interpolating functions!
 *             : Create maxInt and minInt functions
 *             : Change some int-float conversion codes in many functions
 *             : Correct a bug in sumSino when sinoSize is even
 *             : Get rid of sumSino2,4,8 direct2,4,8
 *             : Change the name of the function "bp1" to "direct"
 *
 * New in opt11: Windowed Sinc interpolator
 *
 * The final version is opt11 + some clean-ups.
 * 
 * END MODIFICATION LOG
 */



/*****************************************
Conventions

==============================================
Pixel index

                     |                                      _____
                     |                                     |kT=  |
          ___________|___________                 _____    |  2  |
         |-1.5,|-0.5,| 0.5,| 1.5,|  [N/2-0.5]    |kT=  |   |_____|
         | 1.5 | 1.5 | 1.5 | 1.5 |               | 1.5 |   |kT=  |
         |_____|_____|_____|_____|               |_____|   |  1  |
         |-1.5,|-0.5,| 0.5,| 1.5,|               |kT=  |   |_____|
         | 0.5 | 0.5 | 0.5 | 0.5 |               | 0.5 |   |kT=  |
     ____|_____|_____|_____|_____|____        0__|_____|___|  0  |__0
         |-1.5,|-0.5,| 0.5,| 1.5,|               |kT=  |   |_____|
         |-0.5 |-0.5 |-0.5 |-0.5 |               |-0.5 |   |kT=  |
         |_____|_____|_____|_____|               |_____|   | -1  |
         |-1.5,|-0.5,| 0.5,| 1.5,|               |kT=  |   |_____|
         |-1.5 |-1.5 |-1.5 |-1.5 |               |-1.5 |   |kT=  |
         |_____|_____|_____|_____|  [-N/2+0.5]   |_____|   | -2  |
                     |                            (T=1)    |_____|
      [-N/2+0.5]     |        [N/2-0.5]                     (T=1)
                     |

         |<---------N=4--------->|

===============================================
Sampling

sampling period, T, is the ratio
        
              sampling period of the sinograms
             -----------------------------------
                     size of one pixel

All the sinograms are assumed to be SYMMETRIC around the CENTER (0,0) of the target image.

===============================================
Output Image

Everytime this program runs, it stores the output image in a file named 'outputImage.txt' (in current directory).
The output file contains values of pixels arranged in a way that can be read by MATLAB's load command.

Once in MATLAB, you can view the output image by taking the following steps:

> p = load('outputImage.txt');
> figure, colormap(gray), imagesc(p), colorbar;

This should display the image, in gray scale, in another window.

Note: output filename can be changed. See define section below.
===============================================
Precomputing sine's and cosine's

Before each reconstruction, you must precompute the sine and cosine values for the interested number of angles.
This can be done by calling:

> ./a.out p <# of angles>

for example, if we want to reconstruct an image of size 512x512 using 1024 projections and the filtered sinograms are stored in input.txt, we do the following.

> ./a.out p 1024    <return>
> ./a.out 512 0 input.txt 128 256    <return>

This will reconstruct the image using B=128,D=256, without any debugging messages (the second argument is 0). 

===============================================
END CONVENTIONS
******************************************/


/* Constants */
#define NUM_PARTS 4                       //right now, this program does not support change of this number
#define ERROR_TOLERANCE 0.0001            //EPSILON
#define OUTPUT_FILENAME "outputImage.txt" 
#define TRIG_FILENAME "trig.txt"
//#define float double            //float or double

/* MACROS */

/* Radial Interpolators */
/* !!!IMPORTANT!!! */
/* INTERPOLATING MACROS DO NOT CHECK IF THE INPUT X IS OUT OF BOUND (BEYOND THE SUPPORT) */
/* SO, THE CALLER MUST MAKE SURE (FROM THE SUPPORT INFORMATION) THAT THE INPUTS TO THE MACROS ARE VALID */

#ifndef NEAREST_INTERP
#define NEAREST_INTERP(x) 1                        //the bounds are different from all previous versions (see below)!!!
#endif

#ifndef LINEAR_INTERP
#define LINEAR_INTERP(x) (1.0-fabs(x))
#endif

#ifndef CUBIC_INTERP
#define CUBIC_INTERP(x) (x<ERROR_TOLERANCE &&x>-ERROR_TOLERANCE?1:((x<-1?((x+1)*(x+2)*(x+3)/6):(x<0?(-(x-1)*(x+1)*(x+2)/2):(x<1?((x-2)*(x-1)*(x+1)/2):(-(x-3)*(x-2)*(x-1)/6))))))
#endif

#ifndef CUBIC_SPLINE_INTERP
#define CUBIC_SPLINE_INTERP(x) ((x>1+ERROR_TOLERANCE?(0.2667*((-x+2)*(-x+2)*(-x+2))-0.0667*(-x+1)-0.2667*(-x+2)+0.0667*((-x+1)*(-x+1)*(-x+1))):(x>ERROR_TOLERANCE?(-0.6*((-x+1)*(-x+1)*(-x+1))+0.4*(-x)+1.6*(-x+1)-0.4*(-x*x*x)):(x>-1-ERROR_TOLERANCE?(0.4*(-x*x*x)-1.6*(-x-1)-0.4*(-x)+0.6*(-x-1)*(-x-1)*(-x-1)):(-0.0667*(-x-1)*(-x-1)*(-x-1)+0.2667*(-x-2)+0.0667*(-x-1)-0.2667*(-x-2)*(-x-2)*(-x-2))))))

#endif

#ifndef SINC_INTERP
#define SINC_INTERP(x) (fabs(x)<=ERROR_TOLERANCE?1:sinf(M_PI*x)/x/M_PI*cosf(M_PI*x/6))
#endif

//for each compilation, pick one!

//#define RADIAL_INTERP(x) NEAREST_INTERP(x)
#define RADIAL_INTERP(x) LINEAR_INTERP(x)
//#define RADIAL_INTERP(x) CUBIC_INTERP(x)
//#define RADIAL_INTERP(x) CUBIC_SPLINE_INTERP(x)
//#define RADIAL_INTERP(x) SINC_INTERP(x)


/* Radial Interpolator Supports */
#define NEAREST_SUPPORT 0.4999     //0.5 - ERROR_TOLERANCE
#define LINEAR_SUPPORT 1
#define CUBIC_SUPPORT 2
#define CUBIC_SPLINE_SUPPORT 2
#define SINC_SUPPORT 3

//for each compilation, pick one!

//#define RADIAL_SUPPORT NEAREST_SUPPORT
#define RADIAL_SUPPORT LINEAR_SUPPORT
//#define RADIAL_SUPPORT CUBIC_SUPPORT
//#define RADIAL_SUPPORT CUBIC_SPLINE_SUPPORT
//#define RADIAL_SUPPORT SINC_SUPPORT

/* Angular Downsamplers */
#define PLAIN(x) 1

#define DWNSMPL1(x) (fabs(x)<ERROR_TOLERANCE?0.5:0.25)

//for each compilation, pick one!

//#define ANGULAR_INTERP(x) PLAIN(x)
#define ANGULAR_INTERP(x) DWNSMPL1(x)


/* Angular Downsampler Supports */
#define PLAIN_SUPPORT 0
#define DWNSMPL1_SUPPORT 1

//for each compilation, pick one!

//#define ANGULAR_SUPPORT PLAIN_SUPPORT
#define ANGULAR_SUPPORT DWNSMPL1_SUPPORT


/*Helper functions*/
#define max(x,y) (x>y?x:y)
#define min(x,y) (x<y?x:y)
#define maxInt(x,y) (x>y?x:y)
#define minInt(x,y) (x<y?x:y)

/*END DEFINE*/

float T;
float oneOverT;
int verbose;         //for debugging
float* sine;
float* cosine;
int baseSize;
int dwnsplSize;

typedef struct{
  
  int size;
  float** pixel;                //pixel[0] refers to the lowest row of the image, pixel[i][0] refers to the 
                                  //leftmost column of the image

} image;

typedef struct{

  int num;                       //number of angles
  int size;                      //length of each filtered sinograms
  float T;
  float** sino;
  float* sine;
  float* cosine;

} sinograms;


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
 *
 * takes care of freeing pointers inside image structure
 *
 */
void freeImage(image* img)
{
  int size = img->size;
  int i;
  float** curr = img->pixel;

  for(i=0;i<size;i++){
    free(curr[i]);
  }
}


/*
 * currently, not used.
 * It prints out ascii representation of an image.
 */
void printImage(float** pixel,int size)
{
  int i,j;
  char* table = " .:!+;iot76x0s&8%#@$";
  float max = -10000;
  float min =10000;
  float range;
  int index;

  for(i=size-1;i>=0;i--){
    for(j=0;j<size;j++){
      
      if(max<pixel[i][j]){
	max = pixel[i][j];
      }

      if(min>pixel[i][j]){
	min = pixel[i][j];
      }
      
    }//end for j
  }//end for i

  range = max-min;

  if(range==0){
    range = 1;
  }
 
  for(i=size-1;i>=0;i--){
    for(j=0;j<size;j++){
      index = (floorf((pixel[i][j]-min)/range*20));
      printf("%c ",table[index]);

    }//end for j
    printf("\n");
  }//end for i
}

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
 * Assume DELTA FUNCTION for the SAMPLER.
 * So, basically it just returns intp(m*cos(theta) + n*sin(theta) - (k+tau_p)*T).
 *
 */
float weight(/*kernel* ker,*/float exactRadialPosition,int k)//,float T)
{
  float argToIntp;
  float ans;

  argToIntp = exactRadialPosition-k;     //the unit of argToIntp is in samples
  ans = RADIAL_INTERP(argToIntp);
  
  return ans;
}



/*
 * computes the contributions to a pixel--indexed (m,n)--from all P sinograms
 *
 */
float sumSino(sinograms* g,int m,int n,float* tau,float halfSize,float middle)
{
  int p;
  int k;
  int P=g->num;
  int sinoSize;

  float i;
  float sum=0;
  float* curSino;

  float temp;
  
  float realm=m-halfSize+0.5;     //we need to adjust m,n in order to change them from array indices to 
  float realn=n-halfSize+0.5;     //coordinates on the target image.

  float scaledRealm = realm*oneOverT;
  float scaledRealn = realn*oneOverT;

  float lowerXLimit;
  float upperXLimit;
  float lowerYLimit;
  float upperYLimit;
   
  float spXSupport;
  float spYSupport;
  float intpSupport;

  float curr;
  float lowerKBound;
  float upperKBound;

  int lowerKBound2;
  int upperKBound2;

  float theta;

  float exactRadialPosition;

  intpSupport = RADIAL_SUPPORT;


  sinoSize = g->size;

  //for each sinogram
  for(p=0;p<P;p++){

    curSino = g->sino[p];
    exactRadialPosition = scaledRealm*(g->cosine)[p] + scaledRealn*(g->sine)[p];

    exactRadialPosition -= tau[p]-middle; 
    //exactRadialPosition is now in terms of array index! (although it's still a float)

    //since spSupport is zero
    lowerKBound = exactRadialPosition-intpSupport;
    upperKBound = exactRadialPosition+intpSupport;

    
    lowerKBound2 = maxInt((int)(lowerKBound-ERROR_TOLERANCE)+1,0);
    upperKBound2 = minInt((int)(upperKBound),(g->size)-1);     //use (int) instead of floorf, since upperKBound is a bound for array index which is >=0

    //for nearest interpolator, this needs to be checked
    if(upperKBound2<lowerKBound2){
      upperKBound2 = lowerKBound2;
    }

    //for each entry in this sinogram
    for(k=lowerKBound2;k<=upperKBound2;k++){

      curr = curSino[k];
      
      temp = weight(exactRadialPosition,k);
      sum += curr*temp;

    }//end for k
  }//end for p
	//printf("%f\n",sum);
  return sum;
}


//assume size is a power of 2
/*
 * direct(the sinograms,the size of the image, tau's,output image)
 *
 * direct corresponds EXACTLY to the direct method. 
 *
 */
void direct(sinograms* g,int size,float* tau,image* ans)
{
  int m;        //x direction
  int n;        //y direction
  float temp;

  char* filename = "output.txt";
  FILE* output;

  float* curRow; 
  float middle = ((float)(g->size))/2-0.5;
  float halfSize = 0.5*size;
  int numSino = g->num;
  float scale = M_PI/numSino;       //scale output by PI/(# of projections)  ; see derivation of inverse radon


  //for each pixel in the image
  for(n=0;n<size;n++){
    curRow = (ans->pixel)[n];
    for(m=0;m<size;m++){
      temp = sumSino(g,m,n,tau,halfSize,middle);
      curRow[m] = temp*scale;

    } //end for n
  } //end for m


}

//----------------------------------------------


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
 * caller free both sino and newSino
 * 
 * This function prepares a new set of sinograms for the next recursive call.
 * It DOES angularly downsample the sinograms.
 *
 */
sinograms* newSinoForNextIter(sinograms* sino,int newSinoSize,float constShiftingFactor,float* nu_i)
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
  float shiftingFactor;
  float radialShift;
  float totalShift;
  float temp;
  
  int n2;
  int lowerPBound,upperPBound,lowerKBound,upperKBound;
  float angularSupport;
  float radialSupport;

  //preparing new sinograms
  newSino = (sinograms*)malloc(1*sizeof(sinograms));
  newSino->T = T;
  newSino->size = newSinoSize; 
  newSino->num = ceilf((float)P/2);

  newNumSino = newSino->num;
  (newSino->sino)  = (float**)malloc(newNumSino*sizeof(float*));
  (newSino->sine) = (float*)malloc(newNumSino*sizeof(float));
  (newSino->cosine) = (float*)malloc(newNumSino*sizeof(float));
  //done preparing spaces for new sinograms


  angularSupport = ANGULAR_SUPPORT;
  radialSupport = RADIAL_SUPPORT;


  for(n=0;n<newNumSino;n++){
    //for each new angle
    n2 = 2*n;                
    (newSino->sino)[n] = (float*)malloc(newSinoSize*sizeof(float));
    (newSino->sine)[n] = (sino->sine)[n2];
    (newSino->cosine)[n] = (sino->cosine)[n2];    
    
    lowerPBound = maxInt(n2-(int)(angularSupport),0);    
    upperPBound = minInt(n2+ceilf(angularSupport),P-1);  


    for(m=0;m<newSinoSize;m++){
      //for each new radial array index
      
      sum = 0;
      
      for(p=lowerPBound;p<=upperPBound;p++){
	//for each old angle

	temp1 = n2-p;
	temp2 = ANGULAR_INTERP(temp1);

	radialShift = m+nu_i[n2]-nu_i[p];
	shiftingFactor = rintf(nu_i[p]) + constShiftingFactor;

	lowerKBound = (minInt(maxInt(ceilf(radialShift-shiftingFactor-radialSupport),0),size-1));
	upperKBound = (maxInt(minInt((int)(radialShift-shiftingFactor+radialSupport),size-1),0)); 
	
	//this loop can be optimized more, but i didn't have time.
	for(k=lowerKBound;k<=upperKBound;k++){ 
	  //for each old radial array index
	  temp3 = (sino->sino)[p][k];
	  
	  kk = k + shiftingFactor; 
	  temp = radialShift-kk;
	  temp1 = RADIAL_INTERP(temp);
	  
	  sum += temp1 * temp2 * temp3;

	}//end for k
      }//end for p

      (newSino->sino)[n][m] =  sum;
        
    }//end for m
  }//end for n

  return newSino;
}


/*
 * sino is freed by caller
 * ans is freed by caller
 *
 * This function is the top-level function for recursive backprojection.
 * It makes a call to 'direct' once size<=baseSize.
 */
void bp(sinograms* sino,int size,float* tau,image* ans)
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

  if(size<=baseSize || sino->num <=1){ 
    //BASE CASE!!!!

    direct(sino,size,tau,ans);  

    return;

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
	//for each angle, compute nu_i,p
	
	nu_i[p] = nu(tau[p],shiftX,shiftY,(sino->sine)[p],(sino->cosine)[p]);//,oneOverT);
	nu_temp[p] = rintf(nu_i[p]);
      }//end for p

      //nu_i are real nu's (for this quadrant, for all theta's)
      //nu_temp are rounded nu's

      //decide whether to downsample
      /*  if(size<=dwnsplSize){
	//throw away half the projections as usual
	newSino = newSinoForNextIter(sino,newSinoSize,constShiftingFactor,nu_i);
	pp = 2;   
      } else {  */
	//keep all of them!
	newSino = newSinoForNextIter2(sino,newSinoSize,constShiftingFactor,nu_i);
	pp = 1;   
      //} 

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
      /////////////////////////////////////////////////
      bp(newSino,/*ker,*/newSize,nu_i,subImage);     //
      /////////////////////////////////////////////////


      freeSino(newSino);
      free(newSino);
      newSino = NULL;
      free(nu_i);
      free(nu_temp);
      nu_i = NULL;
    
    }//end for i

    free(subImage);
    free(subImage->pixel);
    //do not free subImage's pixels, because they are the output!
    
    return;

  }//end if not base case


}//end function bp

/*=====================================
Below are old version of interpolating functions.
At first, I used function pointers and passed them around
to keep the interpolator parameterized.
There might be some bugs in them, I cannot guarantee.

They have been replaced with macros. 
=======================================*/

float myRadialIntp0(float x)
{
  if(verbose){
    printf("x=%f | ",x);
  }
  if(x<0.5 && x>=-0.5){
    if(verbose){
      printf("returning 1\n");
    }
    return 1;
  }

  if(verbose){
    printf("returning 0\n");
  }
  return 0;
}

float myRadialIntpSupport0()
{
  return 1;
}

/*
 * linear interpolation
 *
 * CAUTION! this function will not check if x is out of bound (>1 || <-1)
 * to save time. The caller must make sure!
 *
 */
float myRadialIntp1(float x)
{
  if(verbose==10){
    printf("x=%f\n",x);
  }
  if(fabs(x)>=1){
    if(verbose==11){
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n          out of bound!!!!! (x=%f) \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",x);
    }
    return 0;
  }
  return 1-fabs(x);
}

float myRadialIntpSupport1()
{
  return 1;
}

/*
 * Cubic Interpolation
 *
 *  o      *
 *  |\     |---x-->|
 *  | \ _o |
 *  |    |\|_ o____o
 * _|____|_v__|____|___
 *
 *    -->| T  |<--
 */
float myRadialIntp2(float x)
{
  //if x = 0, then the contribution will come from 
  //this data point alone,
  //otherwise all four neighboring points make contributions.
  float oneOverTCube;
  float ans;

  //  printf("x=%f |",x);
  if(x<ERROR_TOLERANCE && x>-ERROR_TOLERANCE){
    //    printf("returning 1 <<\n");
    return 1;
  }  

  if(x>2 || x<-2){
    //    printf("returning 0<<<\n");
    return 0;
  }
  

  if(x<-1){
    //far left
    ans = (x+1)*(x+2)*(x+3)/6;
  } else if(x<0) {
    //middle left
    ans = -(x-1)*(x+1)*(x+2)/2;
  } else if(x<1) {
    //middle right
    ans = (x-2)*(x-1)*(x+1)/2;
  } else {
    //far right
    ans = -(x-3)*(x-2)*(x-1)/6;
  }
    
  //  printf("returning %f\n",ans);
  return ans;
}

float myRadialIntpSupport2()
{
  return 2;
}



/*
 * Natural Cubic Spline (using 3 data points)
 *    
 *    *
 *    |
 *  o-|--o----o
 *  | |  |    |
 *  |_v__|____|
 *
 */
/*
float myRadialIntp3(float x)
{
  float ans;
  //  float flipX = -x;
  
  if(verbose){
    printf("x=%f | ",x);
  }

  if(x<=-1+ERROR_TOLERANCE || x>=2-ERROR_TOLERANCE){
    //out of bound
    if(verbose){
      printf("out of bound\n");
    }
    return 0;
  }

  if(x>1+ERROR_TOLERANCE){
    if(verbose){
      printf("(case1)");
    }
    ans = powf(-x+2,3)/4-(2-x)/4;

  } else if(x>ERROR_TOLERANCE){
    if(verbose){
      printf("(case2)");
    }
    ans = 1.5*(1-x)-powf(1-x,3)/2.0;
    
    if(verbose && x<0.05){
      printf("*+++++++++++++++++++++++++*\n");
      printf("3/2*(1-%f)-(%f)^3 /2 = (%f)-(%f)\n",x,1-x,1.5*(1-x),powf(1-x,3)/2);
    }
  } else {
    //x<=0
    if(verbose){
      printf("(case3)");
    }
    ans = powf(-x,3)/4+1.25*x+1;
  }

  if(verbose){
    printf("returning %f\n",ans);
  }
  return ans;

}


float myRadialIntpSupport3()
{
  return 2;
}
*/

/*
 * Natural Cubic Spline
 * using 4 data points.
 *
 */
float myRadialIntp3(float x)
{
  float ans;
  //  float flipX = -x;
  
  if(verbose){
    printf("x=%f | ",x);
  }
  
  if(x<=-2+ERROR_TOLERANCE || x>=2-ERROR_TOLERANCE){
    //out of bound
    if(verbose){
      printf("out of bound\n");
    }
    return 0;
  }

  if(x>1+ERROR_TOLERANCE){
    if(verbose){
      printf("(case1)");
    }
    ans = 0.2667*powf(-x+2,3)-0.0667*(-x+1)-0.2667*(-x+2)+0.0667*powf(-x+1,3);

  } else if(x>ERROR_TOLERANCE){
    if(verbose){
      printf("(case2)");
    }
    ans = -0.6*powf(-x+1,3)+0.4*(-x)+1.6*(-x+1)-0.4*powf(-x,3);
    
  } else if(x>(-1-ERROR_TOLERANCE)){
    //x<=0
    ans = 0.4*powf(-x,3)-1.6*(-x-1)-0.4*(-x)+0.6*powf(-x-1,3);
    if(verbose){
      printf("(case3)");
      //      printf("0.4*(%f)^3-1.6*(%f)-0.4*(%f)+0.6*(%f)^3 = %f +%f + %f+%f = %f\n",-x-1,-x-2,-x-1,-x-2,-0.0667*powf(-x-1,3),0.4*powf(-x,3)-1.6*(-x-1)-0.4*(-x)+0.6*powf(-x-1,3),ans);
    }


  } else {
    ans = -0.0667*powf(-x-1,3)+0.2667*(-x-2)+0.0667*(-x-1)-0.2667*powf(-x-2,3);
    if(verbose){
      printf("(case4)");
      //printf("-0.0667*(%f)^3+0.2667*(%f)+0.0667*(%f)-0.2667*(%f)^3 = %f +%f + %f+%f = %f\n",-x-1,-x-2,-x-1,-x-2,-0.0667*powf(-x-1,3),0.2667*(-x-2),0.0667*(-x-1),0.2667*powf(-x-2,3),ans);
    }
       
    //   ans = -0.0667*powf(-x-1,3)+0.2667*(-x-2)+0.0667*(-x-1)-0.2667*powf(-x-2,3);
  }

  if(verbose){
    printf("returning %f\n",ans);
  }

  return ans;  
}

float myRadialIntpSupport3()
{
  return 2;
}

float mySincIntp(float x){
  
  float ans;
  
  if(verbose==10){
    printf("x=%f | ",x);
  }

  if(fabs(x)<=ERROR_TOLERANCE){
    if(verbose==10){
      printf(" return 1\n");
    }
    return 1;
  } else {
    ans = sinf(x*M_PI)/x/M_PI*cos(x*M_PI/6);
    
    if(verbose==10){
      printf(" return %f\n",ans);
    }
    return ans;
  }

}

/*
 * plain downsampler
 *
 */
float myAngularIntp(float x)
{

  //  return 1;
  if(fabs(x)<ERROR_TOLERANCE){
    return 0.5;
  } else {
    return 0.25;
  }
  
}

float myAngularIntpSupport()
{
  return 1;
}

float myAngularIntp2(float x)
{
  float T=3;
  if(x>-ERROR_TOLERANCE && x<ERROR_TOLERANCE){
    return 1;
  }

  return sinf(x/T)/x*cosf(x*M_PI/10.0/T);
  
}

float myAngularIntpSupport2()
{
  return 5;
}

/******************************
END old interpolating function section
*******************************/


/*
 * delta function
 * NOT USED ANYMORE
 *
 */
float mySp(float x,float y)
{
  if(fabs(x)<=0.5 && fabs(y)<=0.5){
    return 1;
  } else {
    return 0;
  }
}

/*
 * NOT USED ANYMORE
 */
float mySpXSupport()
{
  return 0;
}

/*
 * NOT USED ANYMORE
 */
float mySpYSupport()
{
  return 0;
}


/*
 * NOT USED ANYMORE
 *  uniform distribution
 *  
 *         theta_p =   (Pi)(p)/P
 *
 */
float myTheta(int p,int P)
{
  return M_PI*p/P+M_PI/2;
}


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
  int tempNumSino;
  int i=0;
  int sinoSize;
  char* curValue = (char*)malloc(25*sizeof(char));
  float curData;
  int j=0;
  int k;
  int m;
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
  //fclose(trigInput);
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
  float theta;

  for(i=0;i<P;i++){
    theta = myTheta(i,P);
    s1[i]=sinf(theta);
    c1[i]=cosf(theta);
	//printf("%f\n",cosf(theta));
  }

  //fclose(trigOutput);
}


int main(int argc,char* argv[])
{
clock_t begin, end;
double time_spent;
//begin = clock();
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
 /*
	//argument3 is either 1 or 0, with 1 meaning 'print out debug statements' 
	//I've deleted almost all of the debug statements, though.
	if(argv[3]==NULL){
		printf("verbose not set\n");
		exit(0);
	}
	verbose = atoi(argv[3]);
  */
	//argument4 is the input filename (containing sinograms)
	if(argv[3]==NULL){
		printf("filename not set\n");
		exit(0);
	}
	strcpy(filename,argv[3]);

	//argument5 is the base case size B.
	if(argv[4] == NULL){
		printf("baseSize not set\n");
		exit(0);
	}

	baseSize = atoi(argv[4]);


															/*Step2 Precompute Trig and Initialize Devices*/
    
	sine = (float*)malloc(numSino*sizeof(float));
	cosine = (float*)malloc(numSino*sizeof(float));
	
	begin = clock();
	precomputeTrig(numSino,sine,cosine);

	end = clock();							
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

  //THE CALL TO DIRECT METHOD
  bp(sino,size,tau,img);

  //output image values to file (for used in matlab)
  if(!(output = fopen(filename2,"w"))){
    printf("Could not open file output.txt\n");
    exit(0);
  }
  

  for(i=size-1;i>=0;i--){
    for(j=0;j<size;j++){
      fprintf(output,"%f ",(img->pixel)[i][j]);  
    }
    fprintf(output,"\n");
    
  }
  //end = clock();
  //time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  //printf("%f\n",time_spent);
  fclose(output);

  
  //time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  //printf("%f\n",time_spent);
  freeSino(sino);
  free(sino);
  freeImage(img);
  free(img);
  free(filename);
  
  return 0;
}
