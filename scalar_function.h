#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"
#include <riscv_vector.h>

#define ll long long


static inline uint64_t read_cycle() {
    uint64_t cycle;
    asm volatile ("rdcycle %0" : "=r" (cycle));
    return cycle;
}


void print_2D_matrix(int * arr,int n,int m){
    int i, j; 
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            printf("%d ",*((arr+i*m) + j));
        }
        printf("\n");
    }
}

void scalar_transpose(long long * a, long long * b,ll n,ll m) 
{ 
    int i, j; 
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            *((b+j*n) + i) = *((a+i*m) + j);
            // co++;
        }
    }
} 