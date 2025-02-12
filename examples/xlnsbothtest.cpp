// this tests that xlns16.cpp and xlns32.cpp can coexist in the same executable using either C calls or C++ overloaded ops 
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define xlns32_ideal
#include "xlns32.cpp"
#define xlns16_ideal
#include "xlns16.cpp"

void test1xlns32()
{
	xlns32 odd,sum,two;
	int i;

	odd = fp2xlns32(1.0);
	sum = fp2xlns32(0.0);
	two = fp2xlns32(2.0);
	for (i=1; i<=10000; i++)
	{
		sum = xlns32_add(sum,odd);
		odd = xlns32_add(odd,two);
	}
	printf("test1xlns32 odd=%f sum=%f\n", xlns322fp(odd),xlns322fp(sum));
}


void test1xlns16()
{
	xlns16 odd,sum,two;
	int i;

	odd = fp2xlns16(1.0);
	sum = fp2xlns16(0.0);
	two = fp2xlns16(2.0);
	for (i=1; i<=10000; i++)
	{
		sum = xlns16_add(sum,odd);
		odd = xlns16_add(odd,two);
	}
	printf("test1xlns16 odd=%f sum=%f\n", xlns162fp(odd),xlns162fp(sum));
}

void test5xlns16_float()
{
	xlns16_float num,val,sum;
	long i;

	num = 1.0;
	sum = 0.0;
	val = num;
	for (i=1; i<=1000; i++)
	{
		sum = sum + val/num;
		val = -val;
		num = num + 2.0;
	}
	std::cout << "test5xlns16_float num=" << num << " 4*sum=" << 4*sum << "\n";
}

void test5xlns32_float()
{
	xlns32_float num,val,sum;
	long i;

	num = 1.0;
	sum = 0.0;
	val = num;
	for (i=1; i<=1000; i++)
	{
		sum = sum + val/num;
		val = -val;
		num = num + 2.0;
	}
	std::cout << "test5xlns32_float num=" << num << " 4*sum=" << 4*sum << "\n";
}

int main()
{
	test5xlns32_float();
	test5xlns16_float();
	test1xlns32();
	test1xlns16();
	return 1;
}

