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
//#include <emmintrin.h> /* SSE2 */
#include "mpi.h"
#include <stdio.h>
#include <string.h>

using namespace std;

string response = "4";

#define N0  784
#define N1  1008
#define N2  512
//we will not be optimizing anything with N3, not multiple of 16 
#define N3 10

struct timeval t1, t2;

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
double localHS2[N2/16];
double localHS1[N1/16];

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

int numprocs, my_id;
int main (int argc, char *argv[]){
memset(HS1, 0, sizeof(HS1));
memset(localHS2, 0, sizeof(localHS2));
MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
MPI_Comm_rank(MPI_COMM_WORLD,&my_id);
cout << "Number " << my_id << " out of " <<numprocs<<endl;
if(my_id ==0){
cout <<"ID 0 is initializing matrices and broadcasting results to all processes"<<endl;
//B1, B2, B3 arrays and W0 W1 W2 matrices populated. 
//Send to all other processes 
//MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,  MPI_Comm comm)
RandomizeWeights();
}

MPI_Bcast(&B1, N1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&B2, N2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&B3, N3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&W0, N0*N1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&W1, N1*N2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&W2, N2*N3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
ParseLabelFile();
ParsePixelFile();
/*
if(my_id ==1){
    cout << " I am 1"<< endl;
    for(int i = 5; i < 8; i++){
    cout << W0[i][i] <<endl;
    }
}
if(my_id ==5){
    cout << " I am 5"<< endl;
      for(int i = 5; i < 8; i++){
    cout << W0[i][i] <<endl;
    }
}
if(my_id ==12){
    cout << " I am 12"<< endl;
      for(int i = 5; i < 8; i++){
    cout << W0[i][i] <<endl;
    }
    
}
*/
if (argc == 2){
    train(atoi(argv[1]));
} 
else {
    train(1000000000);
}
   MPI_Finalize();
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
       double timeA;
    //string response = "10";
   // cout << "Train 10 or 4 Digits?" <<endl;
   // cin >> response;
    if(my_id ==0){
         timeA = MPI_Wtime();
       gettimeofday(&t1, NULL);
    }

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
        if(my_id ==0){
         if(i % 500 == 0){
            OutputStatusToFile(i);
            }
        }
	}
     if(my_id ==0){
         timeA = MPI_Wtime() - timeA; 
         cout << timeA << " MPI GIVEN TIME" << endl;
        gettimeofday(&t2, NULL);
      printf("NN time is %d milliseconds\n",
	       (t2.tv_sec - t1.tv_sec)*1000 + 
	       (t2.tv_usec - t1.tv_usec) / 1000);
     }
 MPI_Barrier(MPI_COMM_WORLD);
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
    double * ptr1 = HS1;
    double Procholder1[N3];
    memset(Procholder1, 0, sizeof(Procholder1));
  //  MPI_Barrier(MPI_COMM_WORLD);
 for (int i = 0; i<N0; i++){ 
		IN[i] = input[i];}
    for (int i=0; i<N1; i++) {
		HS1[i] = B1[i];
	}
  
       // MPI_Barrier(MPI_COMM_WORLD);
         int start, end;
         start = (N1 / 16) *my_id;
         end = (N1 / 16) * (my_id+1);
        int l=0;
        for (int i=start; i<end; i++) { 
 		for (int j=0; j<N0; j++){
			localHS1[l] += IN[j]*W0[j][i];
             }
               l++;
          }
       MPI_Allgather(localHS1, N2/16, MPI_DOUBLE, HS1, N2/16, MPI_DOUBLE, MPI_COMM_WORLD);
          
	   
 
    for (int i=0; i<N1; i++) {
		HO1[i] = alpha(HS1[i]);
	}
      for(int i=0; i<N2;i++){
          HS2[i] = B2[i];
      }
      /*
      for (int i=0; i<N2; i++) {
		for (int j=0; j<N1; j++)
			HS2[i] += HS1[j]*W1[j][i];
	}*/

       start = (N2 / 16) *my_id;
         end = (N2 / 16) * (my_id+1);
      //  MPI_Barrier(MPI_COMM_WORLD);
        int k=0;
      for (int i=start; i<end; i++){
		for (int j=0; j<N1; j++){
			localHS2[k] += HS1[j]*W1[j][i];
          }
          k++;
	    }
    
   // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(localHS2, N2/16, MPI_DOUBLE, HS2, N2/16, MPI_DOUBLE, MPI_COMM_WORLD);   

     for (int i=0; i<N2; i++) {
		HO2[i] = alpha(HS2[i]);
	}
    for(int i=0; i<N3;i++){
          OS[i] = B3[i];
      }
      /*
    for (int i=0; i<N3; i++) {
		for (int j=0; j<N2; j++)
			OS[i] += HS2[j]*W2[j][i];
	}*/

     start = (N2 / 16) *my_id;
        end = (N2 / 16) * (my_id+1);
      //  MPI_Barrier(MPI_COMM_WORLD);

        for (int i=0; i<N3; i++) {
		    for (int j=start; j<end; j++){
			    OS[i] += HS2[j]*W2[j][i];
            }
	    }
    
       //  MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(OS, Procholder1, N3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
           if(my_id == 0){
           for(int i =0; i < N3; i++){
               OS[i] = Procholder1[i];
            }
           }
           MPI_Bcast(OS, N3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

double local_dE_HO2[N2/16];
double local_dE_HO1[N1/16];
double local_dE_W2[N2/16][N3];
double local_dE_W1[N1/16][N2];
double local_dE_W0[N0/16][N1];

double backward(double *O, double *Y){
    int start, end;
    err = 0.0;
	for (int i = 0; i < N3; i++)
		err += (O[i] - Y[i]) * (O[i] - Y[i]);
	err = err / N3;

	// compute dE_OO
	for (int i = 0; i < N3; i++)
		dE_OO[i] = (O[i] - Y[i]) * 2.0 / N3;

	// compute dOO_OS = dAlpha(OS) = AB / cosh^2(B * OS)
	// alpha is our activation function and dAlpha is the derivative of alpha
	for (int i = 0; i < N3; i++)
		dOO_OS[i] = dAlpha(OS[i]);

	// compute dE_OS = dE_OO dot dOO_OS
	for (int i = 0; i < N3; i++)
		dE_OS[i] = dE_OO[i] * dOO_OS[i];

	// compute dE_B3 = dE_OS
	for (int i = 0; i < N3; i++)
		dE_B3[i] = dE_OS[i];

	// compute dE_W2
    
    for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			dE_W2[i][j] = dE_OS[j] * HO2[i];
        /*
        start = (N2 / 16) *my_id;
         end = (N2 / 16) * (my_id+1);
         //int m=0;
	for (int i = start; i < end; i++){
		for (int j = 0; j < N3; j++){
			dE_W2[i][j] = dE_OS[j] * HO2[i];
        }
       // m++;
    }
   // MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Allgather(local_dE_W2, (N2/16)*N3, MPI_DOUBLE, dE_W2, (N2/16)*N3, MPI_DOUBLE, MPI_COMM_WORLD);   
    */
	// compute dE_HO2 = sum_{j = 1}^{N3} dE_OS_i * W2_ij
    start = (N2 / 16) *my_id;
    end = (N2 / 16) * (my_id+1);
    //MPI_Barrier(MPI_COMM_WORLD);
    int k =0;
    for (int i = start; i < end; i++) {
		local_dE_HO2[k] = 0;
		for (int j = 0; j < N3; j++){
			local_dE_HO2[k] += dE_OS[j] * W2[i][j];
            }
        k++;
	}
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(local_dE_HO2, N2/16, MPI_DOUBLE, dE_HO2, N2/16, MPI_DOUBLE, MPI_COMM_WORLD);  
    /*
	for (int i = 0; i < N2; i++) {
		dE_HO2[i] = 0;
		for (int j = 0; j < N3; j++)
			dE_HO2[i] += dE_OS[j] * W2[i][j];
	} */

	// compute dOO_OS = dAlpha(OS) = AB / cosh^2(B * OS)
	// compute dHO2_HS2 = dAlpha(HS2) = AB / cosh^2(B * HS2)
	for (int i = 0; i < N2; i++)
		dHO2_HS2[i] = dAlpha(HS2[i]);

	// compute dE_HS2 = dE_HO2 dot dHO2_HS2
	for (int i = 0; i < N2; i++)
		dE_HS2[i] = dE_HO2[i] * dHO2_HS2[i];

	// compute dE_B2 = dE_HS2
	for (int i = 0; i < N2; i++)
		dE_B2[i] = dE_HS2[i];

	// compute dE_W1
    /*
    for (int i = 0; i < N1; i++)
		for (int j = 0; j < N2; j++)
			dE_W1[i][j] = dE_HS2[j] * HO1[i];
            */
    
     start = (N1 / 16) *my_id;
    end = (N1 / 16) * (my_id+1);
    //int a=0;
	for (int i = start; i < end; i++){
		for (int j = 0; j < N2; j++){
			dE_W1[i][j] = dE_HS2[j] * HO1[i];
        }
       // a++;
    }
   // MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Allgather(local_dE_W1, (N1/16)*N2, MPI_DOUBLE, dE_W1, (N1/16)*N2, MPI_DOUBLE, MPI_COMM_WORLD);  
   
	// compute dH01_HS1
	for (int i = 0; i < N1; i++)
		dHO1_HS1[i] = dAlpha(HS1[i]);


	// compute dE_HO1 = sum_{j = 1}^N2 dE_HS2 * W1_ij
        start = (N1 / 16) *my_id;
         end = (N1 / 16) * (my_id+1);
      //  MPI_Barrier(MPI_COMM_WORLD);
        int l=0;
        for (int i = start; i < end; i++) {
		local_dE_HO1[l] = 0;
		for (int j = 0; j < N2; j++){
			local_dE_HO1[l] += dE_HS2[j] * W1[i][j];
            }
            l++;
        }
   // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(local_dE_HO1, N2/16, MPI_DOUBLE, dE_HO1, N2/16, MPI_DOUBLE, MPI_COMM_WORLD);   
    
    /*
	for (int i = 0; i < N1; i++) {
		dE_HO1[i] = 0;
		for (int j = 0; j < N2; j++)
			dE_HO1[i] += dE_HS2[j] * W1[i][j];
	}
*/
	// compute dE_HS1 = dE_HO1 dot dHO1_HS1
	for (int i = 0; i < N1; i++)
		dE_HS1[i] = dE_HO1[i] * dHO1_HS1[i];

	// compute dE_B1 = dE_HS1
	for (int i = 0; i < N1; i++)
		dE_B1[i] = dE_HS1[i];

	// compute dE_W0
    /*
    for (int i = 0; i < N0; i++)
		for (int j = 0; j < N1; j++)
			dE_W0[i][j] = dE_HS1[j] * IN[i];
            */
    
      start = (N0 / 16) *my_id;
         end = (N0 / 16) * (my_id+1);
        // int b =0;
	for (int i = start; i < end; i++){
		for (int j = 0; j < N1; j++){
			dE_W0[i][j] = dE_HS1[j] * IN[i];
        }
      //  b++;
    }
   // MPI_Barrier(MPI_COMM_WORLD);
   // MPI_Allgather(local_dE_W0, (N0/16)*N1, MPI_DOUBLE, dE_W0, (N0/16)*N1, MPI_DOUBLE, MPI_COMM_WORLD);  

	// update W0, W1, W2, B1, B2, B3
     start = (N0 / 16) *my_id;
         end = (N0 / 16) * (my_id+1);
	for (int i = start; i < end; i++)
		for (int j = 0; j < N1; j++)
			W0[i][j] = W0[i][j] - rate * dE_W0[i][j];

	for (int i = 0; i < N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];

     start = (N1 / 16) *my_id;
    end = (N1 / 16) * (my_id+1);
	for (int i = start; i < end; i++)
		for (int j = 0; j < N2; j++)
			W1[i][j] = W1[i][j] - rate * dE_W1[i][j];

	for (int i = 0; i < N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

	for (int i = 0; i < N2; i++)
		for (int j = 0; j < N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i = 0; i < N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];

}

