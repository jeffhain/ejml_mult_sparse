/*
 * Copyright (c) 2016, Jeff Hain. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ejml.alg.dense.mult;

import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.RowD1Matrix64F;

public class MatrixMatrixMultSparseTest {

    public static void main(String[] args) {
        
        test_computeDensity_RowD1Matrix64F();
        
        test_computeDensity_2RowD1Matrix64F();
        
        test_mustUseMultSparse_2RowD1Matrix64F();
        
        test_mult_sparse_3RowD1Matrix64F_special();
        
        test_mult_sparse_3RowD1Matrix64F_nominal();
        
        /*
         * 
         */
        
        bench_mult_xxx_3RowD1Matrix64F();
    }
    
    /*
     * test
     */

    private static void test_computeDensity_RowD1Matrix64F() {
        for (DenseMatrix64F m : new DenseMatrix64F[]{
                new DenseMatrix64F(0,0),
                new DenseMatrix64F(0,1),
                new DenseMatrix64F(1,0)}) {
            double density = MatrixMatrixMultSparse.computeDensity(m);
            if (!Double.isNaN(density)) {
                throw new AssertionError("" + density);
            }
        }
        for (DenseMatrix64F m : new DenseMatrix64F[]{
                new DenseMatrix64F(1,1),
                new DenseMatrix64F(2,2)}) {
            double density = MatrixMatrixMultSparse.computeDensity(m);
            if (density != 0.0) {
                throw new AssertionError("" + density);
            }
        }
        {
            DenseMatrix64F m = new DenseMatrix64F(2,2);
            m.set(0, 0, 0.1);
            double density = MatrixMatrixMultSparse.computeDensity(m);
            if (density != 0.25) {
                throw new AssertionError("" + density);
            }
        }
        {
            DenseMatrix64F m = new DenseMatrix64F(2,2);
            m.set(0, 0, 0.1);
            m.set(0, 1, 0.2);
            m.set(1, 0, 0.3);
            double density = MatrixMatrixMultSparse.computeDensity(m);
            if (density != 0.75) {
                throw new AssertionError("" + density);
            }
        }
    }

    private static void test_computeDensity_2RowD1Matrix64F() {
        for (DenseMatrix64F m : new DenseMatrix64F[]{
                new DenseMatrix64F(0,0),
                new DenseMatrix64F(0,1),
                new DenseMatrix64F(1,0)}) {
            double density = MatrixMatrixMultSparse.computeDensity(m,m);
            if (!Double.isNaN(density)) {
                throw new AssertionError("" + density);
            }
        }
        for (DenseMatrix64F m : new DenseMatrix64F[]{
                new DenseMatrix64F(1,1),
                new DenseMatrix64F(2,2)}) {
            double density = MatrixMatrixMultSparse.computeDensity(m,m);
            if (density != 0.0) {
                throw new AssertionError("" + density);
            }
        }
        {
            DenseMatrix64F a = new DenseMatrix64F(2,2);
            a.set(0, 0, 0.1);
            DenseMatrix64F b = new DenseMatrix64F(4,4);
            b.set(0, 1, 0.2);
            b.set(1, 0, 0.3);
            double density = MatrixMatrixMultSparse.computeDensity(a,b);
            if (density != (1 + 2) / (double) (2*2 + 4*4)) {
                throw new AssertionError("" + density);
            }
        }
    }
    
    private static void test_mustUseMultSparse_2RowD1Matrix64F() {
        for (DenseMatrix64F m : new DenseMatrix64F[]{
                new DenseMatrix64F(0,0),
                new DenseMatrix64F(0,1),
                new DenseMatrix64F(1,0)}) {
            boolean must = MatrixMatrixMultSparse.mustUseMultSparse(m,m);
            if (must) {
                throw new AssertionError();
            }
        }
        final int nThreshold = (int) Math.sqrt(MatrixMatrixMultSparse.SIZE_THRESHOLD);
        for (DenseMatrix64F m : new DenseMatrix64F[]{
                new DenseMatrix64F(1,1),
                new DenseMatrix64F(nThreshold - 1, nThreshold - 1)}) {
            boolean must = MatrixMatrixMultSparse.mustUseMultSparse(m,m);
            // Too small, must not.
            if (must) {
                throw new AssertionError();
            }
        }
        {
            DenseMatrix64F m = new DenseMatrix64F(100,100);
            final int nbrNonZero = (int) (m.getNumElements() * MatrixMatrixMultSparse.DENSITY_THRESHOLD + 0.5) + 1;
            for (int index = 0; index < nbrNonZero; index++) {
                m.set(index, 1.1);
            }
            boolean must = MatrixMatrixMultSparse.mustUseMultSparse(m,m);
            // Too large density.
            if (must) {
                throw new AssertionError();
            }
        }
        {
            DenseMatrix64F m = new DenseMatrix64F(100,100);
            final int nbrNonZero = (int) (m.getNumElements() * MatrixMatrixMultSparse.DENSITY_THRESHOLD + 0.5);
            for (int index = 0; index < nbrNonZero; index++) {
                m.set(index, 1.1);
            }
            boolean must = MatrixMatrixMultSparse.mustUseMultSparse(m,m);
            // Just small enough density.
            if (!must) {
                throw new AssertionError();
            }
        }
        for (DenseMatrix64F m : new DenseMatrix64F[]{
                new DenseMatrix64F(nThreshold + 1, nThreshold + 1),
                new DenseMatrix64F(1000,1000)}) {
            boolean must = MatrixMatrixMultSparse.mustUseMultSparse(m,m);
            // Large enough and zero density: must.
            if (!must) {
                throw new AssertionError();
            }
        }
    }

    private static void test_mult_sparse_3RowD1Matrix64F_special() {
        DenseMatrix64F a = new DenseMatrix64F(0,0);
        DenseMatrix64F b = new DenseMatrix64F(0,0);
        DenseMatrix64F c = new DenseMatrix64F(0,0);
        MatrixMatrixMultSparse.mult_sparse(a, b, c);
    }

    private static void test_mult_sparse_3RowD1Matrix64F_nominal() {
        
        final Random random = new Random(123456789L);
        
        /*
         * Density and dimensions small enough
         * that we have some zero dot products
         * in output matrix.
         */
        
        double density = 0.5;
        
        DenseMatrix64F a = new DenseMatrix64F(3, 5);
        randomize(random, density, a);
        
        DenseMatrix64F b = new DenseMatrix64F(5, 7);
        randomize(random, density, b);

        DenseMatrix64F refC = new DenseMatrix64F(3, 7);
        // Randomizing output, to make sure even zeros are set.
        randomize(random, refC);
        MatrixMatrixMult.mult_aux(a, b, refC, null);

        DenseMatrix64F c = new DenseMatrix64F(3, 7);
        // Randomizing output, to make sure even zeros are set.
        randomize(random, c);
        MatrixMatrixMultSparse.mult_sparse(a, b, c);

        final String refCStr = refC.toString();
        final String cStr = c.toString();
        
        if (!refCStr.equals(cStr)) {
            System.out.println("refC = " + refCStr);
            System.out.println("c =    " + cStr);
            throw new AssertionError("bad result");
        }
    }
    
    /*
     * bench
     */
    
    /**
     * Also tests that mustUseMultSparse(...) doesn't indicate to use
     * mult_sparse(...) when it's actually slower.
     */
    private static void bench_mult_xxx_3RowD1Matrix64F() {
        
        final Random random = new Random(123456789L);
        
        final int nbrOfRuns = 2;
        
        final double opsMagnitude = 10.0 * 1000.0 * 1000.0;
        
        for (int n = 4; n <= 4096; n *= 2) {

            System.out.println();

            // n^3 because naive multiplication is in O(n^3).
            final int nbrOfCalls = Math.max(1, (int) (opsMagnitude / Math.pow(n, 3)));
            
            final double diagLikeDensity = 1.0/Math.pow(n,2);
            
            // Valid across densities.
            long mult_aux_lastNs = 0L;

            for (double density : new double[]{
                    1.0,
                    0.9,
                    0.75,
                    0.5,
                    0.05,
                    0.005,
                    diagLikeDensity}) {

                DenseMatrix64F a = new DenseMatrix64F(n, n);
                randomize(random, density, a);

                DenseMatrix64F b = new DenseMatrix64F(n, n);
                randomize(random, density, b);

                DenseMatrix64F c = new DenseMatrix64F(n, n);

                if (density != 1.0) {
                    // Same as with density of 1.
                } else {
                    for (int k = 0; k < nbrOfRuns; k++) {
                        long ns1 = System.nanoTime();
                        for (int i = 0; i < nbrOfCalls; i++) {
                            MatrixMatrixMult.mult_aux(a, b, c, null);
                        }
                        long ns2 = System.nanoTime();
                        mult_aux_lastNs = ns2 - ns1;
                        printInfo(nbrOfCalls, "mult_aux(a,b,c,null)", n, density, diagLikeDensity, ns1, ns2);
                    }
                }
                
                if ((n >= 2048)
                        && (density >= 0.5)) {
                    // At this point we know it's slow,
                    // no need to slow down tests further
                    // with these densities.
                    continue;
                }

                long mult_sparse_lastNs = 0L;
                
                for (int k = 0; k < nbrOfRuns; k++) {
                    long ns1 = System.nanoTime();
                    for (int i = 0; i < nbrOfCalls; i++) {
                        MatrixMatrixMultSparse.mult_sparse(a, b, c);
                    }
                    long ns2 = System.nanoTime();
                    mult_sparse_lastNs = ns2 - ns1;
                    printInfo(nbrOfCalls, "  mult_sparse(a,b,c)", n, density, diagLikeDensity, ns1, ns2);
                }
                
                final boolean multSparseExpectedFaster =
                        MatrixMatrixMultSparse.mustUseMultSparse(a, b);
                if (multSparseExpectedFaster) {
                    if (mult_sparse_lastNs > mult_aux_lastNs) {
                        throw new AssertionError(mult_sparse_lastNs + " > " + mult_aux_lastNs);
                    }
                }
                
                // Benching that only when sparse version is better,
                // for checking that density check overhead is still minimal.
                if (multSparseExpectedFaster) {
                    for (int k = 0; k < nbrOfRuns; k++) {
                        long ns1 = System.nanoTime();
                        for (int i = 0; i < nbrOfCalls; i++) {
                            MatrixMatrixMultSparse.mult_smart(a, b, c);
                        }
                        long ns2 = System.nanoTime();
                        mult_sparse_lastNs = ns2 - ns1;
                        printInfo(nbrOfCalls, "   mult_smart(a,b,c)", n, density, diagLikeDensity, ns1, ns2);
                    }
                }
            }
        }
    }
    
    private static void printInfo(
            int nbrOfCalls,
            String methodSignature,
            int n,
            double density,
            double diagLikeDensity,
            long ns1,
            long ns2) {
        System.out.println(
                nbrOfCalls + " calls to " + methodSignature + ", n=" + n
                + ", density=" + ((float) density) + ((density == diagLikeDensity) ? " (diag-like)" : "")
                + " took " + ((ns2-ns1)/1000/1e6) + " s");
    }
    
    /*
     * 
     */

    private static void randomize( Random random, RowD1Matrix64F m ) {
        randomize(random, 1.0, m);
    }

    /**
     * @param density In [0,1].
     */
    private static void randomize( Random random, double density, RowD1Matrix64F m ) {
        
        for ( int i = 0; i < m.numRows; i++ ) {
            for ( int j = 0; j < m.numCols; j++ ) {
                
                if ((density >= 1.0)
                        || (random.nextDouble() < density)) {
                    
                    final double element = random.nextDouble();
                    
                    m.unsafe_set( i, j, element );
                }
            }
        }
    }
}
