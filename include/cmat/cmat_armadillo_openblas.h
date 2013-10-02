//**********
// -cmat_armadillo_openblas.h  
// -Use Armadillo+OpenBLAS as 
//  efficient matrix library
//
// -Piji Li
// *********

#ifndef CMAT_H
#define CMAT_H

#include <stdio.h>
#include <iostream>
#include "armadillo"

using namespace std;
using namespace arma;

template <class T>
class CMat
{
    public:
        CMat();
        CMat(int rows, int cols);
        ~CMat();
        void print();
        void setRow(int rows){row = rows;}
        void setCol(int cols){col = cols;}
        int rows();
        int cols();
        T& operator()(int r, int c);
        CMat<T>& operator=(const CMat<T>& a);
        CMat<T> operator*(const CMat<T>& a);

    private:
        mat data;
        int row;
        int col;
        void destroy();
};

    template <class T>
CMat<T>::CMat()
{
    row = 0;
    col = 0;
}
    template <class T>
CMat<T>::CMat(int rows, int cols)
{
    data = zeros<mat>(rows, cols);
    setRow(rows);
    setCol(cols);
}
    template <class T>
inline void CMat<T>::destroy()
{
}

    template <class T>
CMat<T>::~CMat()
{
    destroy();
}

    template <class T>
inline void CMat<T>::print()
{
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            printf("%f ", (float)data(i, j));
        }
        printf("\n");
    }

}
    template <class T>
inline int CMat<T>::rows()
{
    return row;
}

    template <class T>
inline int CMat<T>::cols()
{
    return col;
}

    template <class T>
inline T& CMat<T>::operator()(int r, int c)
{
    if(r>=0 && c>=0)
        return data(r, c);
}

// assignment
    template <class T>
inline CMat<T>& CMat<T>::operator=(const CMat<T>& a)
{
    setRow(a.row);
    setCol(a.col);

    data = a.data;
    return *this;
}
    template <class T>
inline CMat<T> CMat<T>::operator*(const CMat<T>& b)
{
    if(col == b.row)
    {
        CMat<T> c(row, b.col);
        c.data = data * b.data;
        return c;
    }
    else
        return CMat<T>();
}

#endif
