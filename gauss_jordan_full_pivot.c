/*

This is an implementation of the  Gauss-Jordan Elimination for  solving  linear
sets of equations. If we consider the following set of equations:

    a_00 * x_0     + a_01 * x_1     + ...  + a_0(N-1) * x_(N-1)      =  b_0
    ...
    a_(N-1)0 * x_0 + a_(N-1)1 * x_1 + ...  + a_(N-1)(N-1) * x_(N-1)  =  b_(N-1)

We can write this set of equations in matrix notation as:

    A * x = b,

where A is a matrix and x and b are vectors.  Since the system  has N  unknowns 
and N equations, it is in principle solvable.

For Gauss-Jordan elimination,  we consider  M sets of  the above equations  (in
this example we pick 2). The equation that is solved in this scheme is:

    A * (x0 u x1 u Y) = b0 u b1 u 1,

where A, Y and 1 are N x N matricies, x0, x1, b0, b1 are N dimensional vectors,
'u'  is the column augmentation operator  (stack vectors and matricies to  make 
bigger  matricies), and '*' is the dot product.  After applying the  algorithm,
the x's will be replaced with the solution, and Y will be replaced with the in-
verse of A.

The algorithm works by starting with the 0th row of A, and dividing that row by
a_00. This does not change  the answer as long as we divide the appropriate row
on the right hand side by the same factor.  This is true since we are  just re-
scaling all the coefficients,  but maintains the relationship between them.  We
then substract the  corresponsing amount of the  current row to the other  rows
such that all the values in the 0th column are 0 except for a_00.  We then move
to row 1 and repeat the process with element a_11.

However, the algorithm will unstable for non-trivial reasons.  As a remedy,  we
must consider pivoting, which consists of the following operations:

    1. Exchanging any two rows of A, and exchanging the same corresponding rows
       in the b's and 1. This does not change the result.

    2. Exchanging any two columns of A,  and exchanging the  corresponding rows
       of the x's and the Y. This again does not change the result, but it does
       scramble it, and therefore extra  bookkeeping is required to recover the
       result.

Partial pivoting is when the first operation only is used, and full pivoting is
when both operations are used.  We will implement full pivoting here,  but they
are almost equivalent.

The goal of pivoting is to  put a desirable element  in the diagonal so that it
will be used to divide the rows.  In this case, the element with highest  value
will be considered desirable. Furthermore, since we have already above say ele-
ment a_nn, we can only choose rows n and below, and columns n and right,  since
we don't want to undo our current answer.

*/

#include <stdio.h>


#define FINISHED_NO_ERROR      0
#define SINGULAR_MATRIX_ERROR -1


const int N = 3; // Amount of unknowns.
const int M = 1; // Amount of equations.


inline double fabs(double a) {
    return a >= 0.0 ? a : -a;
}


void swap(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}


int gauss_jordan_full_pivot(double a[N][N], double b[N][M]) {

    int curr_row;
    int curr_col;
    int row_indexes[N];
    int col_indexes[N];
    int pivot[N];

    double biggest;
    double pivot_inverse;
    double dummy;

    for (int i = 0; i < N; i++)
        pivot[i] = 0;

    for (int i = 0; i < N; i++) { // Main loop over the columns to be reduced.

        biggest = 0.0;

        // Search for pivot element.
        for (int row = 0; row < N; row++) {

            if (pivot[row] == 0) { // If pivot[row] != 0, then the row was already used.

                for (int col = 0; col < N; col++) {
                    
                    if (pivot[col] == 0 && (dummy = fabs(a[row][col])) > biggest) {
                        biggest = dummy;
                        curr_row = row;
                        curr_col = col;
                    }
                }
            }
        }

        /*
        We now have a  pivot element,  located at 'row' and 'col'.  The columns
        are not really interchanged, they are just relabeled. If 'row' != 'col'
        then there needs to be an exchange of row.
        */
        
        // For bookkeeping.
        row_indexes[i] = curr_row;
        col_indexes[i] = curr_col;

        // Exchange rows if needed.
        if (curr_row != curr_col) {
            
            for (int col = 0; col < N; col++)
                swap(&a[curr_row][col], &a[curr_col][col]);

            for (int col = 0; col < M; col++)
                swap(&b[curr_row][col], &b[curr_col][col]);

            curr_row = curr_col;
        }

        pivot[curr_row] = 1; // Sets the row and column to 'used'.

        if (a[curr_row][curr_col] == 0.0)
            return SINGULAR_MATRIX_ERROR;

        pivot_inverse = 1.0 / a[curr_row][curr_col];

        for (int col = 0; col < N; col++)
            a[curr_row][col] *= pivot_inverse;
        
        for (int col = 0; col < M; col++)
            b[curr_row][col] *= pivot_inverse;

        for (int row = 0; row < N; row++) {

            if (row != curr_row && (dummy = a[row][curr_col]) != 0.0) {

                for (int col = 0; col < N; col++)
                    a[row][col] -= a[curr_row][col] * dummy;

                for (int col = 0; col < M; col++)
                    b[row][col] -= b[curr_row][col] * dummy;
            }
        }
    }

    // Unscramble the results.
    for (int col = N - 1; col >= 0; col--) {

        if (row_indexes[col] != col_indexes[col]) {
            for (int row = 0; row < N; row++)
                swap(&a[row][row_indexes[col]], &a[row][col_indexes[col]]);
        }
    }

    return FINISHED_NO_ERROR;
}


void show_results(double a[N][N], double b[N][M]) {

    for (int eq = 0; eq < M; eq++) {

        printf("Solved equation set %i:\n", eq + 1);

        for (int row = 0; row < N; row++) {

            for (int col = 0; col < N - 1; col++) {
                printf("%7.4f x%i  + ", a[row][col], col);
            }
            printf("%7.4f x%i  = %7.4f\n", a[row][N-1], N-1, b[row][eq]);
        }
    }
}


int main ()
{
    double a[N][N] = 
    {
        {1,2,3},
        {4,5,6},
        {7,8,9}
    };

    double b[N][M] = 
    {
        {9},
        {5},
        {1}
    };

    int status = gauss_jordan_full_pivot(a, b);

    show_results(a, b);

    return status;
}