#define N (2*2*2*2*2*2+2)  //66
//#define N (2*2*2*2*2*2*2+2)  //130
//#define N (2*2*2*2*2*2*2*2+2)  //258
//#define N (2*2*2*2*2*2*2*2*2+2)  //514

#if N <= 66
    #define BLOCK_SIZE 8
#elif N <= 130
    #define BLOCK_SIZE 16
#elif N <= 258
    #define BLOCK_SIZE 32
#else
    #define BLOCK_SIZE 64
#endif