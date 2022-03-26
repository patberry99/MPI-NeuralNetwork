#include <iostream>
#include <math.h> 
#include <stdlib.h>
#include <string>
#include <fstream>
#include <random>
#include <ctime>
#include<sys/time.h>
#include<time.h>
#include<stdio.h>
#include <unistd.h>
#include <emmintrin.h> /* SSE2 */
//TO COMPILE, add the flag -msse2

using namespace std;

#define N0  784
#define N1  1000
#define N2  500
#define N3 10


//TIME RECORDS 
/////////////////////////////////////////////
// unsigned long long int t1, t2, t3;
struct timespec b, e;
struct timeval t1, t2;

typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
     unsigned a, d;
     asm("cpuid");
     asm volatile("rdtsc" : "=a" (a), "=d" (d));

     return (((ticks)a) | (((ticks)d) << 32));
}

void print_duration(struct timespec *b, struct timespec *c)
{
	long long r = c->tv_nsec - b->tv_nsec;
        r += ((long long)(c->tv_sec - b->tv_sec) ) * 1000000000;
	printf("duration = %lld nanoseconds\n", r);
}
/////////////////////////////////////////////////////////


double IN[N0]; //Input Layer
double W0[N0][N1]; //Weights of Inputs -> Hidden Layer
double B1[N1]; // Hidden Layer 1 Biases 
double HS1[N1]; // Hidden Layer 1 Sums 
double HO1[N1]; //Hidden Layer 1 Outputs
double W1[N1][N2]; //Weights of Hidden 1 Layer -> Hidden 2 Layer 
double B2[N2]; //Biases of Hidden Layer 2
double HS2[N2]; //Hidden Layer 2 Sums
double HO2[N2]; //Hidden Layer 2 Outputs
double W2[N2][N3]; //Weights of Hidden 2 Layer -> Output Layer 
double B3[N3]; //Biases of Output
double OS[N3]; //Output Sums
double OO[N3]; //Output Layer Final Output 

//Helper Function Set
int FlipEndian(int i); //File gives us Big Endian, need to flip to Intel Little Endian
void PrintMatrixImage();
void ConvertVector_Char_to_Double(int iteration);
void ParseLabelFile(); 
void ParsePixelFile(); 
void RandomizeWeights();
void train(int iter);
void forward(double *input);
double backward(double *O, double *Y);
void ActivateInput();
void RandomDistributionVector(int size, double *inputarray);
void RandomDistributionMatrix();
void PopulateY();
void OutputStatusToFile(int i );
bool CheckDigit(string response);

void ParseLabelVector(int iteration);
void ParsePixelVector(int iteration); //Argument is the spceific test case we want to retrieve 

double PixelInput[784];
unsigned char PixelInputChar[60000][784]; //This is a holder variable OF ENTIRE FILE, data will be passed into PixelInput when type converted
double PixelInputActivated[784]; //This will hold the values actually be imported to the forward propogation, utilized paper function

unsigned char CorrectLabelInput_Char;
double CorrectLabelInput;
double Y[N3];

//V3 Array Additions 
unsigned char CorrectLabelInputCharVector[60000];


double err;
double rate = 0.00002; //.0005005;

double A = 1.7159;
double B = 0.6666;
//Activation function 
double alpha(double x)
{
	return A * tanh(B * x);
}

double dAlpha(double x)
{
	return A * B / pow(cosh(B * x), 2);
}

bool FourFlag = false;
int main (int argc, char *argv[]){
RandomizeWeights();
ParseLabelFile();
ParsePixelFile();

if (argc == 2){
    train(atoi(argv[1]));
} 
else {
    train(1000000000);
}
}

int FlipEndian(int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void PrintMatrixImage(){
cout<< "The Current Correct Digit: ";
printf("%u \n", CorrectLabelInput_Char);
int NewLine=0;
 for(int i=0;i<784;++i)
    {
         NewLine++;
        if(PixelInput[i] >200){
            cout << "@";
        }
        else if(PixelInput[i]<200){
            cout << "-";
        }
        if(NewLine == 28){
            NewLine =0;
            cout << endl;
        }
        }
        cout <<endl;
    }


void ConvertVector_Char_to_Double(int iteration){
for(int i=0; i < 784; i++){
unsigned char a = PixelInputChar[iteration][i];
double d = a;
PixelInput[i] = d;
}
}

void ParseLabelFile(){
    string filename = "train-labels-idx1-ubyte";
    ifstream MNIST(filename, ios::binary);
     if (MNIST.is_open())
    {
        int magic_number=0;
        int number_of_labels=0;
        unsigned char a;
        MNIST.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= FlipEndian(magic_number);
        MNIST.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= FlipEndian(number_of_labels);
        for(int i=0; i < number_of_labels; i++){
        MNIST.read((char*)&a,sizeof(a));
        CorrectLabelInputCharVector[i] =a;
       /* if(iteration== i+1){
            CorrectLabelInput_Char = a;
            break;
        }*/

        }
       // CorrectLabelInput = CorrectLabelInput_Char;
    }
}

void ParsePixelFile(){
    string filename = "train-images-idx3-ubyte";
    ifstream MNIST(filename, ios::binary);
     if (MNIST.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        MNIST.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= FlipEndian(magic_number);
        MNIST.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= FlipEndian(number_of_images);
        MNIST.read((char*)&n_rows,sizeof(n_rows));
        n_rows= FlipEndian(n_rows);
        MNIST.read((char*)&n_cols,sizeof(n_cols));
        n_cols= FlipEndian(n_cols);
        for(int i=0; i<number_of_images; i++) 
        {
            for(int r=0;r<n_rows*n_cols;++r)
            {
                    unsigned char temp=0;
                    MNIST.read((char*)&temp,sizeof(temp));
                    PixelInputChar[i][r] = temp;
                    }
        }
        
     }
}

void ParseLabelVector(int iteration){
 CorrectLabelInput_Char = CorrectLabelInputCharVector[iteration];
 CorrectLabelInput = CorrectLabelInput_Char;
}

void ParsePixelVector(int iteration){

    ConvertVector_Char_to_Double(iteration);
}

void RandomDistributionVector(int size, double *inputarray){
        //Type of random number distribution
    std::uniform_real_distribution<double> dist(-.05, .05);  //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng; 
    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}()); 

    // generate 10 random numbers.
    for (int i=0; i<size; i++)
    {
     double hold = dist(rng);
     inputarray[i]= hold;

    }
}
void RandomDistributionMatrix(){
       //Type of random number distribution
    std::uniform_real_distribution<double> dist(-.05, .05);  //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng; 
    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}()); 

    //For W0
    for (int i=0; i< N0; i++)
    {
    for(int j=0; j<N1; j++){
     double hold = dist(rng);
     W0[i][j]= hold;
        }
    }
    //For W1
    for (int i=0; i< N1; i++)
    {
    for(int j=0; j<N2; j++){
     double hold = dist(rng);
     W1[i][j]= hold;
        }
    }
    //For W2
    for (int i=0; i< N2; i++)
    {
    for(int j=0; j<N3; j++){
     double hold = dist(rng);
     W2[i][j]= hold;
        }
    }
}

void RandomizeWeights(){
     RandomDistributionVector(N1,B1);
     RandomDistributionVector(N2,B2);
     RandomDistributionVector(N3,B3);
     RandomDistributionMatrix();
     //Just hard coded the function above for all three weight variables
     //Below were some tests to ensure we are using the [-.05, .05] distribution 
    // cout <<"Post Call W1 " << W0[N0-1][N1-2] <<endl;
    // cout <<"Post Call W2 " << W1[N1-1][N2-2] <<endl;
    // cout <<"Post Call W3 " << W2[N2-1][N3-2] <<endl;   
}
void ActivateInput(){
    for(int i =0; i< 784; i++){
        double hold = PixelInput[i];
        hold = hold / 127.5;
        hold = hold -1;
        PixelInputActivated[i]=hold;
    }
}


 void OutputStatusToFile(int i){
  ofstream ofs2;
  ofs2.open("output.txt", ios_base::app);
  ofs2 << "Iter " << i << ": err =" << err << "\n";
  ofs2 << "The Correct Value: "<< CorrectLabelInput <<endl;
  ofs2 << "Values of OO:" <<endl;
    for(int i=0; i< N3; i++){
        ofs2 << i << ": " << OO[i] <<endl;
        }

  ofstream ofs;
  ofs.open ("weights.txt");
      for (int i=0; i<N1; i++) {
		for (int j=0; j<N0; j++)
			ofs << W0[j][i]<<endl;
	}
    ofs << "-----------------------";
    for (int i=0; i<N2; i++) {
		for (int j=0; j<N1; j++)
			ofs << W1[j][i]<<endl;
	}
     ofs << "-----------------------";
    for (int i=0; i<N3; i++) {
		for (int j=0; j<N2; j++)
			ofs << W2[j][i]<<endl;
	}

 }


bool CheckDigit(string response){
    if(response != "4"){
        return true;
    }
    double CL = CorrectLabelInput;
    if( (CL==4) || (CL ==5) || (CL==6) ||(CL==7) ||(CL==8) ||(CL==9) ){
        return false;
    }
    return true;

}


void train(int iter)
{
    string response;
    cout << "Train 10 or 4 Digits?" <<endl;
    cin >> response;
    //clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b); 
    gettimeofday(&t1, NULL);
	for (int i = 0; i< iter; i++) {
		int ii = i % 60000;
        ParseLabelVector(ii);
        ParsePixelVector(ii);
         if(CheckDigit(response) == false){
             continue;
         }
       // if (i > 100 & (i % 100 == 0)){
       // PrintMatrixImage(); 
       // }
        ActivateInput(); //Fully Tested
    
		forward(PixelInputActivated);
     //   for(int i=0; i <N3; i++){
       //     cout << OO[i] <<endl;
       // }
       PopulateY(); //Fully Tested
       //for(int i=0; i < N3; i++){
        //   cout << " " <<Y[i]<<endl;
       //}
		backward(OO, Y);

	/*if (i > 100 & (i % 100 == 0)){
			cout << "Iter " << i << ": err =" << err << "\n";
            cout << "Values of OO:" <<endl;
            for(int i=0; i< N3; i++){
            cout << i << ": " << OO[i] <<endl;
            }
            }*/
         if(i % 1000 == 0){
            OutputStatusToFile(i);
            
            }
	}
      //clock_gettime(CLOCK_THREAD_CPUTIME_ID, &e);  
      gettimeofday(&t2, NULL);
      printf("NN time is %d milliseconds\n",
	       (t2.tv_sec - t1.tv_sec)*1000 + 
	       (t2.tv_usec - t1.tv_usec) / 1000);

     // print_duration(&b, &e);
}

void PopulateY(){
for(int i=0; i< N3; i++){
    Y[i] = -1.716;
if (i== CorrectLabelInput){
    Y[i] =1.716;
        }
    }
}
void forward(double *input){
     double hold [2];
    __m128d v_hold;
    v_hold = _mm_load_pd(hold);
 for (int i = 0; i<N0; i++)
 { 
		IN[i] = input[i];
 }    
 for (int i = 0; i < N1; i++)
 {
     HS1[i] = B1[i];
 }

 // Locality Optimization Here
   __m128d *v_IN = (__m128d *)IN;
 for (int i = 0; i < N0; i++)
 {
     __m128d *v_HS1 = (__m128d *)HS1;
      __m128d *v_W0 = (__m128d *)W0[i];
     for (int j = 0; j < N1; j += 10)
     {
         // HS1[j] += IN[i] * W0[i][j];
         v_hold = _mm_mul_pd(*v_IN, *v_W0);
         *v_HS1 = _mm_add_pd(v_hold, *v_HS1);
         v_HS1++; v_W0++;
           v_hold = _mm_mul_pd(*v_IN, *v_W0);
         *v_HS1 = _mm_add_pd(v_hold, *v_HS1);
         v_HS1++; v_W0++;
           v_hold = _mm_mul_pd(*v_IN, *v_W0);
         *v_HS1 = _mm_add_pd(v_hold, *v_HS1);
         v_HS1++; v_W0++;
           v_hold = _mm_mul_pd(*v_IN, *v_W0);
         *v_HS1 = _mm_add_pd(v_hold, *v_HS1);
         v_HS1++; v_W0++;
           v_hold = _mm_mul_pd(*v_IN, *v_W0);
         *v_HS1 = _mm_add_pd(v_hold, *v_HS1);
         v_HS1++; v_W0++;
     }
     v_IN++;
 }
for (int i=0; i<N1; i+=5)
{
    HO1[i] = alpha(HS1[i]);
    HO1[i+1] = alpha(HS1[i+1]);
    HO1[i+2] = alpha(HS1[i+2]);
    HO1[i+3] = alpha(HS1[i+3]);
    HO1[i+4] = alpha(HS1[i+4]);
}
      for(int i=0; i<N2;i++){
          HS2[i] = B2[i];
      }

// Locality Optimization Here
   __m128d *v_HS1 = (__m128d *)HS1;
for (int i = 0; i < N1; i++) 
{
    __m128d *v_HS2 = (__m128d *)HS2;
      __m128d *v_W1 = (__m128d *)W1[i];
    for (int j = 0; j < N2; j += 10) 
    {
      //  HS2[j] += HS1[i] * W1[i][j];
       v_hold = _mm_mul_pd(*v_HS1, *v_W1);
         *v_HS2 = _mm_add_pd(v_hold, *v_HS2);
         v_HS2++; v_W1++;
           v_hold = _mm_mul_pd(*v_HS1, *v_W1);
         *v_HS2 = _mm_add_pd(v_hold, *v_HS2);
         v_HS2++; v_W1++;
           v_hold = _mm_mul_pd(*v_HS1, *v_W1);
         *v_HS2 = _mm_add_pd(v_hold, *v_HS2);
         v_HS2++; v_W1++;
           v_hold = _mm_mul_pd(*v_HS1, *v_W1);
         *v_HS2 = _mm_add_pd(v_hold, *v_HS2);
         v_HS2++; v_W1++;
           v_hold = _mm_mul_pd(*v_HS1, *v_W1);
         *v_HS2 = _mm_add_pd(v_hold, *v_HS2);
         v_HS2++; v_W1++;
    }
    v_HS1++;
}
     for (int i=0; i<N2; i++) {
		HO2[i] = alpha(HS2[i]);
	}
    for(int i=0; i<N3;i++){
          OS[i] = B3[i];
      }

    // Locality Optimization Here
    for (int i = 0; i < N2; i++) 
    {
        for (int j = 0; j < N3; j++)
        {
            OS[j] += HS2[i] * W2[i][j];
        }
    }
    for (int i=0; i<N2; i++) {
		OO[i] = alpha(OS[i]);
	}
   
}

double dE_OO[N3];
double dOO_OS[N3];
double dE_OS[N3];
double dE_B3[N3];
double dE_W2[N2][N3];

double dE_HO2[N2];
double dHO2_HS2[N2];
double dE_HS2[N2];
double dE_B2[N2];
double dE_W1[N1][N2];

double dE_HO1[N1];
double dHO1_HS1[N1];
double dE_HS1[N1];
double dE_B1[N1];
double dE_W0[N0][N1];






double backward(double *O, double *Y){

  double ratehold[2];
  ratehold[0] = ratehold[1] = rate;
  __m128d v_ratehold = _mm_load_pd(ratehold);

     err = 0.0;
	for (int i = 0; i < N3; i++){
		err += (O[i] - Y[i]) * (O[i] - Y[i]);
    }
	err = err / N3;

	for (int i = 0; i < N3; i++){
		dE_OO[i] = (O[i] - Y[i]) * 2.0 / N3;
        	dOO_OS[i] = dAlpha(OS[i]);
            dE_OS[i] = dE_OO[i] * dOO_OS[i];
            	dE_B3[i] = dE_OS[i];
    }

	// compute dE_W2
	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			dE_W2[i][j] = dE_OS[j] * HO2[i];

	// compute dE_HO2 = sum_{j = 1}^{N3} dE_OS_i * W2_ij
	for (int i = 0; i < N2; i++) {
		dE_HO2[i] = 0;
		for (int j = 0; j < N3; j++)
			dE_HO2[i] += dE_OS[j] * W2[i][j];
	}
	// compute dOO_OS = dAlpha(OS) = AB / cosh^2(B * OS)
	// compute dHO2_HS2 = dAlpha(HS2) = AB / cosh^2(B * HS2)
	for (int i = 0; i < N2; i++){
		dHO2_HS2[i] = dAlpha(HS2[i]);
        dE_HS2[i] = dE_HO2[i] * dHO2_HS2[i];
        dE_B2[i] = dE_HS2[i];
    }

	// compute dE_W1
	for (int i = 0; i < N1; i+=20) {
         __m128d *v_dE_HS2 = (__m128d *)dE_HS2;
		for (int j = 0; j < N2; j+=20){
            for(int ii=i; ii<(i+20); ii++){
                 __m128d *v_dE_W1 = (__m128d *)dE_W1[ii];
                  double hold [2];
                 hold[0] = hold[1] = HO1[ii];
                __m128d v_hold;
                v_hold = _mm_load_pd(hold);
                for(int jj=j; jj<(j+20); jj++){
			       // dE_W1[ii][jj] = dE_HS2[jj] * HO1[ii];
                     *v_dE_W1 = _mm_mul_pd(v_hold, *v_dE_HS2);
                    v_dE_W1++; v_dE_HS2++;
               }
            }
        }
	}
	
	// compute dE_HO1 = sum_{j = 1}^N2 dE_HS2 * W1_ij
       __m128d *v1_dE_HO1 = (__m128d *)dE_HO1;
	for (int i = 0; i < N1; i++) {
         __m128d *v_W1 = (__m128d *)W1[i];
         __m128d *v_dE_HS2 = (__m128d *)dE_HS2;
		for (int j = 0; j < N2; j+=10){
			//dE_HO1[i] = dE_HS2[j] * W1[i][j];
             *v1_dE_HO1 = _mm_mul_pd(*v_dE_HS2, *v_W1);
             v_W1++; v_dE_HS2++;
             *v1_dE_HO1 = _mm_mul_pd(*v_dE_HS2, *v_W1);
             v_W1++; v_dE_HS2++;
             *v1_dE_HO1 = _mm_mul_pd(*v_dE_HS2, *v_W1);
             v_W1++; v_dE_HS2++;
             *v1_dE_HO1 = _mm_mul_pd(*v_dE_HS2, *v_W1);
             v_W1++; v_dE_HS2++;
             *v1_dE_HO1 = _mm_mul_pd(*v_dE_HS2, *v_W1);
             v_W1++; v_dE_HS2++;
        }
        v1_dE_HO1++;
	}

	// compute dE_HS1 = dE_HO1 dot dHO1_HS1

	for (int i = 0; i < N1; i++){
        dHO1_HS1[i] = dAlpha(HS1[i]);}

        //SSE Optimize Below 
        //dE_HS1[i] = dE_HO1[i] * dHO1_HS1[i];
        //This one is good 
          __m128d *v_dE_HO1 = (__m128d *)dE_HO1;
         __m128d *v_dHO1_HS1 = (__m128d *)dHO1_HS1;
         __m128d *v_dE_HS1 = (__m128d *)dE_HS1;
          for (int i = 0; i < N1; i+=10){
            *v_dE_HS1= _mm_mul_pd(*v_dE_HO1, *v_dHO1_HS1);
            v_dHO1_HS1++;
            v_dE_HO1++;
            v_dE_HS1++;
             *v_dE_HS1= _mm_mul_pd(*v_dE_HO1, *v_dHO1_HS1);
            v_dHO1_HS1++;
            v_dE_HO1++;
            v_dE_HS1++;
             *v_dE_HS1= _mm_mul_pd(*v_dE_HO1, *v_dHO1_HS1);
            v_dHO1_HS1++;
            v_dE_HO1++;
            v_dE_HS1++;
             *v_dE_HS1= _mm_mul_pd(*v_dE_HO1, *v_dHO1_HS1);
            v_dHO1_HS1++;
            v_dE_HO1++;
            v_dE_HS1++;
             *v_dE_HS1= _mm_mul_pd(*v_dE_HO1, *v_dHO1_HS1);
            v_dHO1_HS1++;
            v_dE_HO1++;
            v_dE_HS1++;
          }

         for (int i = 0; i < N1; i++){
        dE_B1[i] = dE_HS1[i];
    }

 
for (int i = 0; i < N0; i++){
    		//	dE_W0[i][j] = dE_HS1[j] * IN[i];
     double hold [2];
     hold[0] = hold[1] = IN[i];
       __m128d v_hold;
       v_hold = _mm_load_pd(hold);
     __m128d *v_dE_W0 = (__m128d *)dE_W0[i];
      __m128d *v_dE_HS1 = (__m128d *)dE_HS1;
		for (int j = 0; j < N1; j+=10){
                *v_dE_W0 = _mm_mul_pd(v_hold, *v_dE_HS1);
                v_dE_W0++; v_dE_HS1++;
                 *v_dE_W0 = _mm_mul_pd(v_hold, *v_dE_HS1);
                v_dE_W0++; v_dE_HS1++;
                 *v_dE_W0 = _mm_mul_pd(v_hold, *v_dE_HS1);
                v_dE_W0++; v_dE_HS1++;
                 *v_dE_W0 = _mm_mul_pd(v_hold, *v_dE_HS1);
                v_dE_W0++; v_dE_HS1++;
                 *v_dE_W0 = _mm_mul_pd(v_hold, *v_dE_HS1);
                v_dE_W0++; v_dE_HS1++;
        }
    }
  
	// update W0, W1, W2, B1, B2, B3

	for (int i = 0; i < N0; i++){
         double hold[2];
        __m128d *v_W0 = (__m128d *)W0[i];
        __m128d *v_dE_W0 = (__m128d *)dE_W0[i];
         __m128d v_hold = _mm_load_pd(hold);
		for (int j = 0; j < N1; j+=10){
           // W0[i][j] = W0[i][j] - rate * dE_W0[i][j];
             v_hold = _mm_mul_pd(v_ratehold, *v_dE_W0);
              *v_W0 = _mm_sub_pd(*v_W0, v_hold);
               v_dE_W0++; v_W0++;
               v_hold = _mm_mul_pd(v_ratehold, *v_dE_W0);
              *v_W0 = _mm_sub_pd(*v_W0, v_hold);
               v_dE_W0++; v_W0++;
               v_hold = _mm_mul_pd(v_ratehold, *v_dE_W0);
              *v_W0 = _mm_sub_pd(*v_W0, v_hold);
               v_dE_W0++; v_W0++;
               v_hold = _mm_mul_pd(v_ratehold, *v_dE_W0);
              *v_W0 = _mm_sub_pd(*v_W0, v_hold);
               v_dE_W0++; v_W0++;
               v_hold = _mm_mul_pd(v_ratehold, *v_dE_W0);
              *v_W0 = _mm_sub_pd(*v_W0, v_hold);
               v_dE_W0++; v_W0++;
        }
    }
   

     //Above is replaced with SIMD SSE
     //REDO - look at the example 
     //B1[i] = B1[i] - rate * dE_B1[i];
        double hold[2];
        __m128d *v_B1 = (__m128d *)B1;
        __m128d *v_dE_B1 = (__m128d *)dE_B1;
         __m128d v_hold = _mm_load_pd(hold);
        for(int i =0; i < N1; i+=10){
            v_hold = _mm_mul_pd(v_ratehold, *v_dE_B1);
            *v_B1 = _mm_sub_pd(*v_B1, v_hold);
            v_B1++;
            v_dE_B1++;
              v_hold = _mm_mul_pd(v_ratehold, *v_dE_B1);
            *v_B1 = _mm_sub_pd(*v_B1, v_hold);
            v_B1++;
            v_dE_B1++;
              v_hold = _mm_mul_pd(v_ratehold, *v_dE_B1);
            *v_B1 = _mm_sub_pd(*v_B1, v_hold);
            v_B1++;
            v_dE_B1++;
              v_hold = _mm_mul_pd(v_ratehold, *v_dE_B1);
            *v_B1 = _mm_sub_pd(*v_B1, v_hold);
            v_B1++;
            v_dE_B1++;
              v_hold = _mm_mul_pd(v_ratehold, *v_dE_B1);
            *v_B1 = _mm_sub_pd(*v_B1, v_hold);
            v_B1++;
            v_dE_B1++;
        }

	for (int i = 0; i < N1; i++){
        double hold[2];
        __m128d *v_W1 = (__m128d *)W1[i];
        __m128d *v_dE_W1 = (__m128d *)dE_W1[i];
         __m128d v_hold = _mm_load_pd(hold);
		for (int j = 0; j < N2; j+=10){
			//W1[i][j] = W1[i][j] - rate * dE_W1[i][j];
           v_hold = _mm_mul_pd(v_ratehold, *v_dE_W1);
            *v_W1 = _mm_sub_pd(*v_W1, v_hold);
            v_dE_W1++; v_W1++;
             v_hold = _mm_mul_pd(v_ratehold, *v_dE_W1);
            *v_W1 = _mm_sub_pd(*v_W1, v_hold);
            v_dE_W1++; v_W1++;
             v_hold = _mm_mul_pd(v_ratehold, *v_dE_W1);
            *v_W1 = _mm_sub_pd(*v_W1, v_hold);
            v_dE_W1++; v_W1++;
             v_hold = _mm_mul_pd(v_ratehold, *v_dE_W1);
            *v_W1 = _mm_sub_pd(*v_W1, v_hold);
            v_dE_W1++; v_W1++;
             v_hold = _mm_mul_pd(v_ratehold, *v_dE_W1);
            *v_W1 = _mm_sub_pd(*v_W1, v_hold);
            v_dE_W1++; v_W1++;
        }
    }

   	//B2[i] = B2[i] - rate * dE_B2[i];
       double hold1[2];
        __m128d *v_B2 = (__m128d *)B2;
        __m128d *v_dE_B2 = (__m128d *)dE_B2;
         __m128d v_hold1 = _mm_load_pd(hold1);
        for(int i =0; i < N2; i+=10){
            v_hold1 = _mm_mul_pd(v_ratehold, *v_dE_B2);
            *v_B2 = _mm_sub_pd(*v_B2, v_hold1);
            v_B2++;
            v_dE_B2++;
             v_hold1 = _mm_mul_pd(v_ratehold, *v_dE_B2);
            *v_B2 = _mm_sub_pd(*v_B2, v_hold1);
            v_B2++;
            v_dE_B2++;
             v_hold1 = _mm_mul_pd(v_ratehold, *v_dE_B2);
            *v_B2 = _mm_sub_pd(*v_B2, v_hold1);
            v_B2++;
            v_dE_B2++;
             v_hold1 = _mm_mul_pd(v_ratehold, *v_dE_B2);
            *v_B2 = _mm_sub_pd(*v_B2, v_hold1);
            v_B2++;
            v_dE_B2++;
            v_hold1 = _mm_mul_pd(v_ratehold, *v_dE_B2);
            *v_B2 = _mm_sub_pd(*v_B2, v_hold1);
            v_B2++;
            v_dE_B2++;
        }

	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i = 0; i < N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];

}
