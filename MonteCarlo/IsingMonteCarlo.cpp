// A program to simulate the Ising and (1,d-1) codes on a D-dimensional Hypercube
// Roger Melko, June 8, 2013
//
// Requires BOOST multi_array: http://www.boost.org
// compile example:  g++ -std=c++11 IsingMonteCarlo.cpp -I /opt/local/include/ -o IsingMonteCarlo
//


#include <iostream> 
#include <vector> 
#include <time.h>
using namespace std;

#include <boost/multi_array.hpp>

#include "hypercube.h"
#include "MersenneTwister.h"
#include "simparam.h"
#include "generalD_1_2.code.h"
//#include "isingHamiltonian.h"
#include "measure.h"
#include "percolation.h"

int main ( int argc, char *argv[] )
{
    int start_time = time(NULL);
    int seed_add;
    if ( argc != 2 ){ 
        //cout<<"usage: "<< argv[0] <<" integer \n";
        //return 1;
        seed_add = 0;
    }
    else {
        seed_add = strtol(argv[1], NULL, 10);
    }

    //First, we call several constructors for the various objects used
    PARAMS param; //read parameter file: L, D, T, etc.  See param.data
    MTRand mrand(param.SEED_+seed_add); //random number generator

    HyperCube cube(param.nX_,param.Dim_); //initialize the lattice

    //define the Ising variables +1 or -1 
    Spins sigma; //Assign number of spins in the Hamiltonian below

    //IsingHamiltonian hamil(sigma,cube); //Ising model
    GeneralD12Code hamil(sigma,cube,param.H_); //toric code

    //hamil.PreparePercolation(sigma,cube); //for D>2 toric code percolation only

    //Percolation perc(hamil.N_); //Ising model
    //Percolation perc(hamil.N2); //Toric code

    //perc.DetermineClusters(hamil.All_Neighbors,hamil.occupancy); //Ising
    //perc.DetermineClusters(hamil.TwoCellNeighbors,hamil.occupancy);  //Toric code
    //perc.print();

    //Measure accum(hamil.N_,param);     //Ising model
    Measure accum(hamil.N1,param);  //toric code

    double H = param.H_;
    double T = param.Temp_;
    int counter = 0;
    //This is the temperature loop
    for (T = param.Temp_; T<param.Tlow_; T+=param.Tstep_){ //down
        //Equilibriation
        for (int i=0; i<param.EQL_; i++) {
            hamil.LocalUpdate(sigma,T,mrand,H);
            hamil.GaugeUpdate(sigma,T,mrand,H);
        }
        //MCS binning
        for (int k=0; k<param.nBin_; k++){ 
            accum.zero();
            //perc.zero();
            for (int i=0; i<param.MCS_; i++){ 
                hamil.LocalUpdate(sigma,T,mrand,H);
                hamil.GaugeUpdate(sigma,T,mrand,H);
                //hamil.CalculateOccupancy(sigma); //now calculated in the LocalUpdate
                //perc.DetermineClusters(hamil.All_Neighbors,hamil.occupancy); //Ising
                //perc.DetermineClusters(hamil.TwoCellNeighbors,hamil.occupancy); //Toric code
                accum.record(hamil.Energy,sigma,hamil.WilsonLoops);
				//accum.outputWilsonLoop(sigma,hamil.WilsonLoops,seed_add);

            }//i
            accum.output(T,H,seed_add);
            accum.save(H, param.Dim_, param.nX_, T, seed_add, k);
            //perc.output(T,param.MCS_);
            //hamil.print(sigma);
            sigma.save(param.Dim_, param.nX_, T, seed_add, k);
            counter++;
            if ((counter)%50==0){
                int current_time = time(NULL) - start_time;
                int hours = current_time / 3600;
                int minutes = (current_time % 3600) / 60;
                int seconds = (current_time % 3600) % 60;
                cout << "Saved " << counter << "th lattice after " << hours << " h " << minutes << " m " << seconds << " s" << endl;
            }
        }//k
    }//T

    return 0;

}
