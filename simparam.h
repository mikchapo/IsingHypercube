#ifndef PARAMSIM_H /* Start of header stuff */
#define PARAMSIM_H

#include <fstream> /* File inout/output and console input and output */
#include <iostream>
using namespace std;

//Simple class to read in the simulation parameters from a file
class PARAMS
{   /* described parameters, all straightforward */
    public:
        int nX_;      //linear size of lattice
        int Dim_;      //dimension of lattice
        double H_;      //Zeeman field
        double Temp_;      //Temperature limit 1
        double Tlow_;      //Temperature limit 2
        double Tstep_;      //Temperature step
        int EQL_;     //the number of equilibration steps
        int MCS_;     //the number of Monte Carlo steps
        int nBin_;    //number of production bins
        long SEED_;   //the random number seed

        PARAMS();
        void print();


}; //PARAMS
/* Initializes all parameters */
PARAMS::PARAMS(){
    //initializes commonly used parameters from a file
    ifstream pfin;
    pfin.open("param.dat");

    pfin >> nX_;
    pfin >> Dim_;
    pfin >> H_;
    pfin >> Temp_;
    pfin >> Tlow_;
    pfin >> Tstep_;
    pfin >> EQL_;
    pfin >> MCS_;
    pfin >> nBin_;
    pfin >> SEED_;
    pfin.close();

}//constructor

/* Print the parameters */
void PARAMS::print(){

    cout<<"Linear size "<<nX_<<endl;
    cout<<"Dimension "<<Dim_<<endl;
    cout<<"Magnetic Field"<<H_<<endl;
    cout<<"# Equil steps "<<EQL_<<endl;
    cout<<"# MC steps "<<MCS_<<endl;
    cout<<"# data bins "<<nBin_<<endl;
    cout<<"RNG seed "<<SEED_<<endl;

}

#endif
