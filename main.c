#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"
#include <riscv_vector.h>

#include "scalar_function.h"
#include "vector_function.h"

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

#define ll long long
#define BLOCK_SIZE 4000  // Adjust this value based on the cache size and vector length
#define ROW_BLOCK_SIZE 10



int main() {

    ll n = 4;
    ll m = 50;


    ll input[m] __attribute__((aligned(32*NR_LANES)));
    ll a[m][n] __attribute__((aligned(32*NR_LANES)));
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            a[i][j] = (i + j) % 10 + 1;
        }
    }

    for(int i = 0; i < m; i++) {
        input[i] = i % 10+1;
    }


    ll transpose_matrix[n][m];


    ll output[n] __attribute__((aligned(32*NR_LANES)));
    ll scalar_output[n] __attribute__((aligned(32*NR_LANES)));

    ll v1 = get_cycle_count();

    rvv_matrix_transpose(transpose_matrix[0],a[0],m,n);
    solve(input, output, transpose_matrix[0], n, m);
    
    ll v2 = get_cycle_count();

    ll vector_time = v2 - v1;


    memset(transpose_matrix, 0, sizeof(transpose_matrix));


    ll s1 = get_cycle_count();
    
    scalar_transpose(&a[0][0],&transpose_matrix[0][0],m,n);


    for(int i = 0; i < n; i++) {
        ll t = 0;
        for(int j = 0; j < m; j++) {
            t += transpose_matrix[i][j] * input[j];
        }
        scalar_output[i] = t;
    }

    ll s2 = get_cycle_count();

    ll scalar_time = s2 - s1;
    

    printf("\n");
    printf("\n");

    
    for(int i = 0; i < n; i++) {

        if(scalar_output[i] != output[i]) {
            printf("Error: vector answer not correct\n");
            return 0;
        }
    }


    printf("Scalar performance: %lld cycles\n", scalar_time);
    printf("Vector performance: %lld cycles\n", vector_time);
    printf("Speedup: %lldx\n", scalar_time / vector_time);

    return 0;
}
