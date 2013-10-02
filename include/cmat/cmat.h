//**********
// cmat.h 
// Matrix 
// Piji Li
// *********

#ifndef CMAT_H
#define CMAT_H

#include <stdio.h>
#include <iostream>

using namespace std;
//namespace pgmat
//{
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
        T **data;
        int row;
        int col;
        void destroy();
};

    template <class T>
CMat<T>::CMat()
{
    data = NULL;
    row = 0;
    col = 0;
}
    template <class T>
CMat<T>::CMat(int rows, int cols)
{
    setRow(rows);
    setCol(cols);

    data = new T*[row];
    for(int i = 0; i < row; ++i)
    {
        data[i] = new T[col];
    }
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            data[i][j] = 0;
        }
    }

}
    template <class T>
inline void CMat<T>::destroy()
{
    //for(int i=0; i<row; i++)
     //   delete[] data[i];
    //delete[] data;
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
            printf("%f ", (float)data[i][j]);
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
        return data[r][c];
}

// assignment
    template <class T>
inline CMat<T>& CMat<T>::operator=(const CMat<T>& a)
{
    setRow(a.row);
    setCol(a.col);

    data = new T*[row];
    for(int i = 0; i < row; ++i)
    {
        data[i] = new T[col];
    }
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            data[i][j] = a.data[i][j];
        }
    }
    return *this;
}
    template <class T>
inline CMat<T> CMat<T>::operator*(const CMat<T>& b)
{
    //assert(a.col == b.row);

    if(col == b.row)
    {    
        CMat<T> c(row, b.col);
        
        for(int i=0; i<row; i++)
        {
            
            for(int k=0; k<b.col; k++)
            {
                for(int j=0; j<b.col; j++)
                {
                    c.data[i][k] += data[i][j] * b.data[j][k];
                }
            }
        }
        return c;
    }
    else
        return CMat<T>();
}

//}

#endif
