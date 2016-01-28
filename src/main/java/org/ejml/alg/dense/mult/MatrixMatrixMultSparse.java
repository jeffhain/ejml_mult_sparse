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

import java.util.Arrays;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.ops.MatrixDimensionException;

/**
 * Principal method is mult_sparse(...), for fast multiplication of sparse
 * RowD1Matrix64F matrices.
 * Only useful above a certain size, and below a certain density.
 */
public class MatrixMatrixMultSparse {

    private static final double GROWTH_FACTOR = 1.5;

    /**
     * With JDK8, sparse algo is better more often.
     * Being conservative so that it's better even with older JDKs.
     */
    private static final boolean HAVE_JDK8_PERFS = false;
    
    static final double SIZE_THRESHOLD = HAVE_JDK8_PERFS ? 20.0 * 20.0 : 100.0 * 100.0;

    static final double DENSITY_THRESHOLD = HAVE_JDK8_PERFS ? 0.1 : 0.06;

    /**
     * Cf. ArrayList.
     * 
     * The maximum size of array to allocate.
     * Some VMs reserve some header words in an array.
     * Attempts to allocate larger arrays may result in
     * OutOfMemoryError: Requested array size exceeds VM limit
     */
    private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
    
    /*
     * 
     */

    /**
     * To help decide whether sparse multiplication is preferable
     * to a more naive algorithm.
     * 
     * @return The density, or NaN for a matrix with no element.
     */
    public static double computeDensity( RowD1Matrix64F m ) {
        final int size = m.getNumElements();
        
        int nonZeroCount = 0;
        final double[] data = m.getData();
        for (int k = m.getNumElements(); --k >= 0;) {
            // Not bothering with counting NaNs here,
            // to make it faster, since there is no
            // NaN propagation to ensure.
            if (data[k] != 0.0) {
                ++nonZeroCount;
            }
        }
        
        return nonZeroCount / (double) size;
    }
    
    /**
     * @return The density of {a U b}.
     */
    public static double computeDensity( RowD1Matrix64F a, RowD1Matrix64F b ) {
        
        final double aDensity = computeDensity(a);
        final double bDensity = computeDensity(b);
        
        // Cast to double to avoid overflow.
        final int aSize = a.getNumElements();
        final int bSize = b.getNumElements();
        final double totalSize = aSize + (double) bSize;
        
        return (aSize * aDensity + bSize * bDensity) / totalSize;
    }

    /**
     * @return True if mult_sparse should surely be about same speed or faster
     *         than mult_aux(...), false otherwise.
     */
    public static boolean mustUseMultSparse( RowD1Matrix64F a , RowD1Matrix64F b )
    {
        // Double because at this point it might overflow.
        final double resultSize = a.numRows * (double) b.numCols;
        return (resultSize >= SIZE_THRESHOLD)
                && (computeDensity(a, b) <= DENSITY_THRESHOLD);
    }

    /*
     * 
     */

    /**
     * Uses mult_sparse if mustUseMultSparse(...) returns true,
     * uses mult_aux(...) otherwise.
     */
    public static void mult_smart( RowD1Matrix64F a , RowD1Matrix64F b , RowD1Matrix64F c )
    {
        if (mustUseMultSparse(a, b)) {
            mult_sparse(a, b, c);
        } else {
            MatrixMatrixMult.mult_aux(a, b, c, null);
        }
    }
    
    /*
     * 
     */

    /**
     * Efficient for matrices above a certain size, and with small density,
     * not efficient for small matrices with non-small density.
     */
    public static void mult_sparse( RowD1Matrix64F a , RowD1Matrix64F b , RowD1Matrix64F c )
    {
        if( a == c || b == c )
            throw new IllegalArgumentException("Neither 'a' or 'b' can be the same matrix as 'c'");
        else if( a.numCols != b.numRows ) {
            throw new MatrixDimensionException("The 'a' and 'b' matrices do not have compatible dimensions");
        } else if( a.numRows != c.numRows || b.numCols != c.numCols ) {
            throw new MatrixDimensionException("The results matrix does not have the desired dimensions");
        }
        
        /*
         * Since we only set non-zero elements of 'c',
         * we need to fill it with 0.0 before.
         */
        
        Arrays.fill(c.getData(), 0, c.getNumElements(), 0.0);

        /*
         * Computing non-zero elements of columns, in a cache-friendly way
         * (with respect to 'b' array, not to our (hopefully little) SparseAux'es
         * internal arrays).
         * But even if this part causes cache-misses, it shouldn't hurt much.
         * NB: Since we don't compute column after column, we'll need to get rid
         * of empty instances afterwards, when we can figure them out, so that the
         * loop on 'b' columns doesn't iterate on them for each row of 'a'.
         */
        
        final SparseAux[] bColAuxAll = new SparseAux[b.numCols];
        for( int j = 0; j < b.numCols; j++ ) {
            bColAuxAll[j] = new SparseAux(j);
        }
        for( int i = 0; i < b.numRows; i++ ) {
            for( int j = 0; j < b.numCols; j++ ) {
                final double element = b.unsafe_get(i, j);
                if (isNonZeroOrIsNaN(element)) {
                    final SparseAux bColAux = bColAuxAll[j];
                    bColAux.add(i, element);
                }
            }
        }

        /*
         * Only keeping non-empty instances.
         * Preserving order, in case it could help for sets into output array.
         */
        
        final SparseAux[] bColAuxArr = new SparseAux[b.numCols];
        int bColAuxArrSize = 0;
        for( int j = 0; j < b.numCols; j++ ) {
            final SparseAux bColAux = bColAuxAll[j];
            if (bColAux.size != 0) {
                bColAuxArr[bColAuxArrSize] = bColAux;
                ++bColAuxArrSize;
            }
            bColAuxAll[j] = null; // No more needed.
        }

        final SparseAux aRowAux = new SparseAux();
        
        /*
         * Each round of first loop is cache-friendly with respect to 'a' and 'c',
         * since it deals with only one row of each.
         */
        
        for( int i = 0; i < a.numRows; i++ ) {
            
            aRowAux.reset(i);
            for( int j = 0; j < a.numCols; j++ ) {
                final double element = a.unsafe_get(i, j);
                if (isNonZeroOrIsNaN(element)) {
                    aRowAux.add(j, element);
                }
            }
            
            if (aRowAux.size == 0) {
                // Empty row.
                continue;
            }
            
            /*
             * It seems having 'a' row indexes already in cache and in proper order
             * helps dot product be less slow for high densities < 1, which tend
             * to cause harmful random memory access patterns.
             * Loading elements array as well doesn't seem to help more.
             */
            
            loadIndexArrInMemory(aRowAux);
            
            /*
             * Only iterating on non-empty columns of 'b'.
             */
            
            for( int k = 0; k < bColAuxArrSize; k++ ) {
                final SparseAux bColAux = bColAuxArr[k];
                final int j = bColAux.index;

                final double element = computeDotProduct(
                        aRowAux.indexArr,
                        bColAux.indexArr,
                        aRowAux.elementArr,
                        bColAux.elementArr,
                        aRowAux.size,
                        bColAux.size);
                
                c.unsafe_set( i, j, element );
            }
        }
    }
    
    /*
     * 
     */
    
    private static void loadIndexArrInMemory(SparseAux aux) {
        double dummy = 0.0;
        for( int k = 0; k < aux.size; k++ ) {
            dummy += aux.indexArr[k];
        }
        if (dummy == Math.PI+Math.E) {
            // Can't happen since we added integers.
            if (Math.sin(dummy) == 0.0) {
                // Can't happen either, but JIT should not bother to figure it out.
                throw new AssertionError("WAT?");
            }
        }
    }
    
    /*
     * 
     */
    
    /**
     * @return True if the specified element is != 0.0,
     *         or is NaN (not ignoring NaNs to ensure NaN propagation).
     */
    private static boolean isNonZeroOrIsNaN(double element) {
        return !(element == 0.0);
    }
    
    /**
     * Computes dot-product by iterating only on non-zero elements.
     */
    private static double computeDotProduct(
            int[] aRowJByAk,
            int[] bColIByBk,
            double[] aElementByAk,
            double[] bElementByBk,
            int aRowSize,
            int bColSize) {

        double sum = 0.0;
        
        int ak = 0;
        int bk = 0;
        while (true) {
            final int aj = aRowJByAk[ak];
            final int bi = bColIByBk[bk];
            if (aj == bi) {
                sum += aElementByAk[ak] * bElementByBk[bk];
                if ((++ak == aRowSize)
                        || (++bk == bColSize)) {
                    break;
                }
            } else if (aj < bi) {
                if (++ak == aRowSize) {
                    break;
                }
            } else {
                if (++bk == bColSize) {
                    break;
                }
            }
        }
        
        return sum;
    }
    
    /*
     * 
     */

    private static class SparseAux {
        private static final int[] EMPTY_INT_ARR = new int[0];
        private static final double[] EMPTY_DOUBLE_ARR = new double[0];
        int index;
        int[] indexArr = EMPTY_INT_ARR;
        double[] elementArr = EMPTY_DOUBLE_ARR;
        int size;
        public SparseAux() {
        }
        public SparseAux(int index) {
            this.index = index;
        }
        @Override
        public String toString() {
            return "[index=" + index
                    + ", size = " + size
                    + ", indexArr = " + Arrays.toString(this.indexArr)
                    + ", elementArr = " + Arrays.toString(this.elementArr)
                    + "]";
        }
        /**
         * Sets size to zero.
         * 
         * @param index Index in the matrix of the row or column of which
         *        this instance is the sparse representation.
         */
        public void reset(int index) {
            this.index = index;
            this.size = 0;
        }
        /**
         * @param index Index of the specified element.
         * @param element Element at the specified index, typically non-zero.
         */
        public void add(int index, double element) {
            this.ensureCapacity(this.size + 1);
            this.indexArr[this.size] = index;
            this.elementArr[this.size] = element;
            ++this.size;
        }
        public void ensureCapacity(int minCapacity) {
            final int oldCapacity = this.indexArr.length;
            if (oldCapacity >= minCapacity) {
                // Already large enough.
                return;
            }
            
            int newCapacity = Math.min(MAX_ARRAY_SIZE, (int) (GROWTH_FACTOR * oldCapacity));
            if (newCapacity < minCapacity) {
                // At least what user asks for (at his own risk).
                newCapacity = minCapacity;
            }
            
            // Created together, for better chance of being close in memory.
            final int[] newIndexArr = new int[newCapacity];
            final double[] newElementArr = new double[newCapacity];
            
            System.arraycopy(this.indexArr, 0, newIndexArr, 0, this.size);
            System.arraycopy(this.elementArr, 0, newElementArr, 0, this.size);
            
            // Set together, to make sure we don't blow up in-between,
            // corrupting the state.
            this.indexArr = newIndexArr;
            this.elementArr = newElementArr;
        }
    }
}
