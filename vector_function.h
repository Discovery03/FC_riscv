#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runtime.h"
#include <riscv_vector.h>


#define ll long long
#define BLOCK_SIZE 4000  // Adjust this value based on the cache size and vector length
#define ROW_BLOCK_SIZE 10

void solve(int64_t *input, int64_t *output, int64_t *a, int64_t curr, int64_t m) __attribute__((always_inline));

void solve(int64_t *input, int64_t *output, int64_t *a, int64_t n, int64_t m) {

    for (int row_block_start = 0; row_block_start < n; row_block_start += ROW_BLOCK_SIZE) {
        int row_block_end = (row_block_start + ROW_BLOCK_SIZE > n) ? n : row_block_start + ROW_BLOCK_SIZE;

        for (int i = row_block_start; i < row_block_end; i++) {
            int64_t *t2 = a + i * m;

            int64_t result = 0;

            for (int block_start = 0; block_start < m; block_start += BLOCK_SIZE) {
                int block_end = (block_start + BLOCK_SIZE > m) ? m : block_start + BLOCK_SIZE;

                int64_t *t1 = input + block_start;
                int64_t *t2_block = t2 + block_start;
                int64_t sz = block_end - block_start;
                int64_t t = block_end - block_start;
                vint64m8_t res;
                int vl;

                int first_iteration = 1;

                while (sz > 0) {
                    vl = vsetvl_e64m8(sz);
                    vint64m8_t a1 = vle64_v_i64m8(t1, vl);
                    vint64m8_t a2 = vle64_v_i64m8(t2_block, vl);

                    if (first_iteration) {
                        res = vmul_vv_i64m8(a1, a2, vl);
                        first_iteration = 0;
                    } else {
                        res = vmacc_vv_i64m8(res, a1, a2, vl);
                    }

                    sz -= vl;
                    t1 += vl;
                    t2_block += vl;
                }

                int si = vsetvl_e64m8(t);
                vint64m1_t red = vmv_s_x_i64m1(red,0, si);
                red = vredsum_vs_i64m8_i64m1(red, res, red, si);
                int64_t r = vmv_x_s_i64m1_i64(red);
                result += r;
            }

            output[i] = result;
        }
    }
}

void rvv_matrix_transpose(long long *dst,long long *src,size_t n,size_t m) 
{
    if(m >= n){
        for (size_t i = 0; i < n; ++i) { 
            size_t avl = m;
            long long* row_src = src + i * m;
            long long* row_dst = dst + i;
            // int co = 0;
            while (avl > 0) {
                size_t vl = vsetvl_e64m8(avl);
                vint64m8_t row = vle64_v_i64m8(row_src, vl);
                vsse64(row_dst, sizeof(long long) * n, row, vl);
                avl -= vl;
                row_src += vl;
                row_dst += vl * n;
                // co++;
            }
        
        }
    }else{

        for (size_t i = 0; i < m; ++i) { 

            size_t avl = n;
            long long* col_src = src + i;
            long long* row_dst = dst + i * n;

            while (avl > 0) {
                size_t vl = vsetvl_e64m8(avl);
                vint64m8_t row = vlse64_v_i64m8(col_src, sizeof(long long) * m,vl);
                vse64(row_dst, row, vl);
                avl -= vl;
                col_src += vl * m;
                row_dst += vl;
            }

        }
    }

}



