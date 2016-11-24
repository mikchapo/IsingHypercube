#ifndef SPINS_H
#define SPINS_H

// spins.h
// a small class that contains a vector of lattice Ising spins

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "MersenneTwister.h"

using namespace std;

class Spins
{
    public:
        int N_; //total number of lattice sites

        //the lattice is a vector of vectors: no double counting
        vector<int> spin;

        //public functions
        Spins(int N);
        Spins();
        void resize(int N);
        void flip(int index);
        void print();
        void randomize();
        void save(int Dim, int L, double T, long seed, int k);

};

//constructor 1
//takes the total number of lattice sites
Spins::Spins(){

    spin.clear(); 

}

//constructor 2
//takes the total number of lattice sites
Spins::Spins(int N){

    N_ = N;

    spin.resize(N_,1); //assign every spin as 1

}

//takes the total number of lattice sites
void Spins::resize(int N){

    N_ = N;

    spin.resize(N_,1); //assign every spin as 1

}

void Spins::randomize(){

    MTRand irand(129345); //random number 

    int ising_spin;
    for (int i = 0; i<spin.size(); i++){
        ising_spin = 2*irand.randInt(1)-1;
        //cout<<ising_spin<<" ";
        spin.at(i) = ising_spin;
    }


}//randomize

//a single-spin flip
void Spins::flip(int index){

    spin.at(index) *= -1;

}//flip


//a print function
void Spins::print(){

    for (int i=0;i<spin.size();i++){
        cout<<(spin[i]+1)/2<<" ";
    }//i
    cout<<endl;

}//print

//a save function
void Spins::save(int Dim, int L, double T, long seed, int k) {
    string filename = "/data/TC-D0-L00-Raw/T00000-00-00000";
    string path = "..";
    string Dim_s, L_s, T_s, seed_s, k_s;
    Dim_s = std::to_string(Dim);
    L_s = std::to_string(L);
    // int T_f = int(1000.0 * T + 0.5);
    // T_f = int(T_f);
    T_s = std::to_string(T);
    seed_s = std::to_string(seed);
    k_s = std::to_string(k);
    filename[10] = Dim_s[0];
    
    if (seed > 9) {
        filename[27] = seed_s[0];
        filename[28] = seed_s[1];
    } else {
        filename[28] = seed_s[0];
    }

    // if (T_f > 999) {
    //     filename[20] = T_s[0];
    //     filename[21] = T_s[1];
    //     filename[22] = T_s[2];
    //     filename[23] = T_s[3];
    // } else if (T_f > 99) {
    //     filename[21] = T_s[0];
    //     filename[22] = T_s[1];
    //     filename[23] = T_s[2];
    // } else if (T_f > 9) {
    //     filename[22] = T_s[0];
    //     filename[23] = T_s[1];
    // } else {
    //     filename[23] = T_s[0];
    // }

    if (T > 9999) {
        filename[21] = T_s[0];
        filename[22] = T_s[1];
        filename[23] = T_s[2];
        filename[24] = T_s[3];
        filename[25] = T_s[4];
    } else if (T > 999) {
        filename[22] = T_s[0];
        filename[23] = T_s[1];
        filename[24] = T_s[2];
        filename[25] = T_s[3];
    } else if (T > 99) {
        if (T_s.find('.') == string::npos) {
            filename[23] = T_s[0];
            filename[24] = T_s[1];
            filename[25] = T_s[2];
        } else {
            filename[21] = T_s[0];
            filename[22] = T_s[1];
            filename[23] = T_s[2];
            filename[24] = ',';
            filename[25] = T_s[4];
        }
    } else if (T > 9) {
        if (T_s.find('.') == string::npos) {
            filename[24] = T_s[0];
            filename[25] = T_s[1];
        } else {
            filename[21] = T_s[0];
            filename[22] = T_s[1];
            filename[23] = ',';
            filename[24] = T_s[3];
            filename[25] = T_s[4];
        }
    } else {
        if (T_s.find('.') == string::npos) {
            filename[25] = T_s[0];
        } else {
            filename[21] = T_s[0];
            filename[22] = ',';
            filename[23] = T_s[2];
            filename[24] = T_s[3];
            filename[25] = T_s[4];
        }
    }
    
    if (L<10){
        filename[14] = L_s[0];
    } else {
        filename[13] = L_s[0];
        filename[14] = L_s[1];
    }
    
    if (k>9999) {
        filename[30] = k_s[0];
        filename[31] = k_s[1];
        filename[32] = k_s[2];
        filename[33] = k_s[3];
        filename[34] = k_s[4];
    } else if (k>999) {
        filename[31] = k_s[0];
        filename[32] = k_s[1];
        filename[33] = k_s[2];
        filename[34] = k_s[3];
    } else if (k>99) {
        filename[32] = k_s[0];
        filename[33] = k_s[1];
        filename[34] = k_s[2];
    } else if (k>9) {
        filename[33] = k_s[0];
        filename[34] = k_s[1];
    } else {
        filename[34] = k_s[0];
    }
    
    filename = path + filename;

    cout << filename << '\n';

    ofstream file;
    file.open(filename,ios::app);
    for (int i=0; i<spin.size(); i++) {
        if (i != (spin.size()-1)) {
            file << spin[i] << ",";
        } else {
            file << spin[i];
        }
    };
    file.close();
}//save

// void Spins::print(){
//     int P1;
//     for (int i=0;i<spin.size();i++){
//         if (i%20<9){
//             if (i<19) {
//                 P1 = spin[2*(i%20) + 1] * spin[2*(i%20) + 3] * spin[2*(i%20)] * spin[180 + 2*(i%20)];
//             } else {
//                 P1 = spin[20*(i/20) + 2*(i%20) + 1] * spin[20*(i/20) + 2*(i%20) + 3] * spin[20*(i/20) + 2*(i%20)] * spin[20*(i/20 -1) + 2*(i%20)];
//             }
//             if (P1==1) {
//                 cout << (spin[20*(i/20) + 2*(i%20) + 1] + 1)/2 << " " << "\033[1;32m"<< P1 << "\033[0m" << " ";
//             } else {
//                 cout << (spin[20*(i/20) + 2*(i%20) + 1] + 1)/2 << "\033[1;31m"<< P1 << "\033[0m" << " ";
//             }
//         } else if (i%20==9) {
//             if (i==9) {
//                 P1 = spin[2*(i%20) + 1] * spin[1] * spin[2*(i%20)] * spin[180 + 2*(i%20)];
//             } else {
//                 P1 = spin[20*(i/20) + 2*(i%20) + 1] * spin[20*(i/20) + 2*(i%20) -17] * spin[20*(i/20) + 2*(i%20)] * spin[20*(i/20 -1) + 2*(i%20)];
//             }
//             if (P1==1) {
//                 cout << (spin[20*(i/20) + 2*(i%20) + 1] + 1)/2 << " " << "\033[1;32m"<< P1 << "\033[0m" << " ";
//             } else {
//                 cout << (spin[20*(i/20) + 2*(i%20) + 1] + 1)/2 << "\033[1;31m"<< P1 << "\033[0m" << " ";
//             }
//         } else if (i%20==10) {
//             cout << "\u00B7 " <<(spin[20*(i/20) + 2*((i-10)%20)] + 1)/2 << " \u00B7 ";
//         } else if (i%20<19) {
//             cout << (spin[20*(i/20) + 2*((i-10)%20)] + 1)/2 << " \u00B7 ";
//         } else {
//             cout << (spin[20*(i/20) + 2*((i-10)%20)] + 1)/2 << "   ";
//         }
//         if ((i+1)%10==0){
//             cout<<endl;
//         }
//     }//i
//     cout<<endl;

// }//print

#endif
