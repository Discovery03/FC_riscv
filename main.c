#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"
#include <riscv_vector.h>

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif



void solve(int64_t *input, int64_t *output, int64_t *a, int64_t curr, int64_t m) __attribute__((always_inline));

void solve(int64_t *input, int64_t *output, int64_t *a, int64_t curr, int64_t m) {

    int64_t *t1 = input;
    int64_t *t2 = a;
    int64_t sz = m;
    vint64m8_t res;
    int vl;
    
    while(sz > 0) {

        vl = vsetvl_e64m8(sz);
        vint64m8_t a1 = vle64_v_i64m8(t1, vl);
        vint64m8_t a2 = vle64_v_i64m8(t2, vl);
        if(sz == m) {
            res = vmul_vv_i64m8(a1, a2, vl);
        } else {
            res = vmacc_vv_i64m8(res, a1, a2, vl);
        }
        sz -= vl;
        t1 += vl;
        t2 += vl;

    }

    vint64m1_t red = vmv_s_x_i64m1(red, 0, vsetvl_e64m8(m)); 
    red = vredsum_vs_i64m8_i64m1(red, res, red, vsetvl_e64m8(m));
    int64_t r = vmv_x_s_i64m1_i64(red);
    output[curr] = r;

}

int main() {

    int64_t n = 3;
    int64_t m = 1000;


    int64_t input[m] __attribute__((aligned(32*NR_LANES)));
    int64_t a[n][m] __attribute__((aligned(32*NR_LANES)));
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            a[i][j] = (i + j) % 10;
        }
    }

    for(int i = 0; i < m; i++) {
        input[i] = i % 10;
    }

    int64_t output[n] __attribute__((aligned(32*NR_LANES)));
    int64_t scalar_output[n] __attribute__((aligned(32*NR_LANES)));

    int64_t v1 = get_cycle_count();
    for(int i = 0; i < n; i++) {
        solve(input, output, a[i], i, m);
    }
    int64_t v2 = get_cycle_count();

    int64_t vector_time = v2 - v1;

    int64_t s1 = get_cycle_count();

    for(int i = 0; i < n; i++) {

        scalar_output[i] = 0;

        for(int j = 0; j < m; j++) {
            scalar_output[i] += a[i][j] * input[j];
        }
    }


    int64_t s2 = get_cycle_count();

    int64_t scalar_time = s2 - s1;


    for(int i = 0; i < n; i++) {

        if(scalar_output[i] != output[i]) {
            printf("Error: vector answer not correct\n");
            return 0;
        }
    }

    printf("Scalar performance: %ld cycles\n", scalar_time);
    printf("Vector performance: %ld cycles\n", vector_time);
    printf("Speedup: %ldx\n", scalar_time / vector_time);

    return 0;
}
