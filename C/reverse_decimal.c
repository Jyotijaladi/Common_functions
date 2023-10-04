#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

void GetFloattoInt (double fnum, long precision, long *pe, long *pd)
{
  long pe_sign;
  long intpart;
  float decpart;

  if(fnum>=0)
  {
    pe_sign=1;
  }
  else
  {
    pe_sign=-1;
  }

  intpart=(long)fnum;
  decpart=fnum-intpart;

  *pe=intpart;  
  *pd=(((long)(decpart*pe_sign*pow(10,precision)))%(long)pow(10,precision));
}


double  reverseDecimal(double  number) {

    int integerPart =  (int)number;
    float decimalPart = number - integerPart;
    double swappedNumber = decimalPart + integerPart;

    
    return swappedNumber;
}