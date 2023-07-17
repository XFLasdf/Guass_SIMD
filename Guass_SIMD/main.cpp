#include <iostream>
#include <nmmintrin.h>
#include <windows.h>
#include <stdlib.h>


using namespace std;

//矩阵规模
const int N = 1000;

//系数矩阵
//float **m;
float m[N][N];

void m_reset(int);
void m_gauss(int);
void m_gauss_simd(int);
void m_gauss_simd_div(int);
void m_gauss_simd_mul(int);
void m_gauss_simd_align(int);

int main()
{
   // N=100;
   // m=new float*[N];
   // for(int i=0;i<N;i++)
    //    m[i]=new float[N];

   // m_reset();

   // for(int i=0;i<N;i++)
   //     delete[]m[i];
   // delete[]m;
    long long head, tail , freq ; // timers
    int step = 10;

    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );

    for(int n = 10; n <= 1000; n += step)
    {
        cout << "问题规模n: " << n << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        m_gauss(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout << "串行算法时间：" << ( tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        m_gauss_simd_div(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout << "除法部分向量化、不对齐时间：" << ( tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        m_gauss_simd_mul(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout << "乘法部分向量化、不对齐时间：" << ( tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        m_gauss_simd(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout << "全部向量化、不对齐时间：" << ( tail - head) * 1000.0 / freq << "ms" << endl;

        m_reset(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        m_gauss_simd_align(n);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout << "全部向量化、对齐时间：" << ( tail - head) * 1000.0 / freq << "ms" << endl;


        if(n == 100) step = 100;
    }



    return 0;
}

//初始化矩阵元素
void m_reset(int n)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<i;j++)
            m[i][j]=0;
        m[i][ i]=1.0;
        for(int j=i+1;j<n;j++)
            m[i][j]=rand();
    }
    for(int k=0;k<n;k++)
        for(int i=k+1;i<n;i++)
            for(int j=0;j<n;j++)
                m[i][j]+=m[k][j];
}

//串行普通高斯消去算法
void m_gauss(int n)
{
    for(int k = 0 ; k < n ; k++)
    {
        for(int j = k+1 ; j < n ; j++)
        {
            m[k][j] = m[k][j]/m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k+1 ; i < n ; i++)
        {
            for(int j = k+1 ; j < n ; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;

        }
    }
}

//除法部分向量化、不对齐
void m_gauss_simd_div(int n)
{
    __m128 vt, va;
    for(int k = 0; k < n; k++){
        vt = _mm_set_ps1(m[k][k]);
        int j;
        for(j = k+1; j+4 <= n; j+=4){
            va = _mm_loadu_ps(&m[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&m[k][j], va);
        }
        if(j < n){
            for(;j < n; j++){
                m[k][j] = m[k][j]/m[k][k];
            }
        }
        m[k][k] = 1.0;
        for(int i = k+1 ; i < n ; i++)
        {
            for(int j = k+1 ; j < n ; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;

        }
    }
}

//乘法部分向量化、不对齐
void m_gauss_simd_mul(int n)
{
    __m128 vaik, vakj, vaij, vx;
    for(int k = 0; k < n; k++){
        for(int j = k+1 ; j < n ; j++)
        {
            m[k][j] = m[k][j]/m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k+1; i < n; i++){
            vaik = _mm_set_ps1(m[i][k]);
            int j;
            for(j = k+1; j+4 <= n; j+=4){
                vakj = _mm_loadu_ps(&m[k][j]);
                vaij = _mm_loadu_ps(&m[i][j]);
                vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&m[i][j], vaij);
            }
            if(j < n){
                for(;j < n; j++){
                    m[i][j] = m[i][j] - m[k][j]*m[i][k];
                }
            }
            m[i][k] = 0;
        }
    }
}

//乘法和除法部分全部向量化、不对齐
void m_gauss_simd(int n)
{
    __m128 vt, va, vaik, vakj, vaij, vx;
    for(int k = 0; k < n; k++){
        vt = _mm_set_ps1(m[k][k]);
        int j;
        for(j = k+1; j+4 <= n; j+=4){
            va = _mm_loadu_ps(&m[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&m[k][j], va);
        }
        if(j < n){
            for(;j < n; j++){
                m[k][j] = m[k][j]/m[k][k];
            }
        }
        m[k][k] = 1.0;
        for(int i = k+1; i < n; i++){
            vaik = _mm_set_ps1(m[i][k]);
            int j;
            for(j = k+1; j+4 <= n; j+=4){
                vakj = _mm_loadu_ps(&m[k][j]);
                vaij = _mm_loadu_ps(&m[i][j]);
                vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&m[i][j], vaij);
            }
            if(j < n){
                for(;j < n; j++){
                    m[i][j] = m[i][j] - m[k][j]*m[i][k];
                }
            }
            m[i][k] = 0;
        }
    }
}

//乘法和除法部分全部向量化、对齐
void m_gauss_simd_align(int n)
{
    __m128 vt, va, vaik, vakj, vaij, vx;
    for(int k = 0; k < n; k++){
        vt = _mm_set_ps1(m[k][k]);
        int j;
        int start = k-k%4+4;
        for(j = k+1; j < start && j < n; j++){
            m[k][j] = m[k][j]/m[k][k];
        }
        if(j != n){
            for(j = start; j+4 <= n; j+=4){
                va = _mm_load_ps(&m[k][j]);
                va = _mm_div_ps(va, vt);
                _mm_store_ps(&m[k][j], va);
            }
            if(j < n){
                for(;j < n; j++){
                    m[k][j] = m[k][j]/m[k][k];
                }
            }
        }
        m[k][k] = 1.0;
        for(int i = k+1; i < n; i++){
            vaik = _mm_set_ps1(m[i][k]);
            int j;
            int start = k-k%4+4;
            for(j = k+1; j < start && j < n; j++){
                m[i][j] = m[i][j] - m[k][j]*m[i][k];
            }
            if(j != n){
                for(j = start; j+4 <= n; j+=4){
                    vakj = _mm_load_ps(&m[k][j]);
                    vaij = _mm_load_ps(&m[i][j]);
                    vx = _mm_mul_ps(vakj, vaik);
                    vaij = _mm_sub_ps(vaij, vx);
                    _mm_store_ps(&m[i][j], vaij);
                }
                if(j < n){
                    for(;j < n; j++){
                        m[i][j] = m[i][j] - m[k][j]*m[i][k];
                    }
                }
            }
            m[i][k] = 0;
        }
    }
}
