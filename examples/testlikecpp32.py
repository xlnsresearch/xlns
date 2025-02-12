import xlns as xl

def test1fp():
	odd = 1
	sum = 0
	for i in range(1,10001):
		sum += odd
		odd += 2.0
	print("test1fp odd="+str(odd)+" sum="+str(sum))

def test2fp():
	num = 1.0
	fact = 1.0
	sum = 0.0
	for i in range(1,9):
		sum = sum + 1.0/fact
		fact = fact * num
		num = num + 1.0
	print("test2fp1 num="+str(num)+" fact="+str(fact)+" sum="+str(sum))

def test3fp():
	num = 1.0
	fact = 1.0
	sum = 0.0
	for i in range(1,11):
		sum = sum + 1.0/fact
		fact = -fact * num * (num + 1.0)
		num = num + 2.0
	print("test3fp1 num="+str(num)+" fact="+str(fact)+" sum="+str(sum))



# compute pi the hard way

def test5xlns32_float():
	num = xl.xlnsv(1.0, 23)
	sum = xl.xlnsv(0.0, 23)
	val = xl.xlnsv(num, 23)
	for i in range(1,1001):
		sum = sum + val/num
		val = -val
		num = num + 2.0
	print("test5xlns32_float num="+str(num)+" 4*sum="+str(4*sum))

def test5fp():
	num = 1.0
	sum = 0.0
	val = num
	for i in range(1,1001):
		sum = sum + val/num
		val = -val
		num = num + 2.0
	print("test5fp num="+str(num)+" 4*sum="+str(4*sum))

#Mandelbrot set fp versus xlns (much slower than compiled cpp due to Python class overhead)

def test4fp(iter):
	mone = -1.0
	two = 2.0
	four = 4.0
	yscale = 12.0
	xscale = 24.0
	for iy in range(11,-12,-1):
		for ix in range(-40, 39):
			y = iy/yscale
			x = ix/xscale
			x1 = x
			y1 = y
			count = 0
			while ((x*x+y*y < four) and (count<iter)):
				xnew = x*x - y*y + x1
				ynew = x*y*two + y1
				count += 1
				x = xnew
				y = ynew
			if (count < iter):
				print("*",end="")
			else:
				print(" ",end="")
		print("")

def test4xlns32_float(iter):
	mone = xl.xlnsv(-1.0, 23)
	two = xl.xlnsv(2.0, 23)
	four = xl.xlnsv(4.0, 23)
	yscale = xl.xlnsv(12.0, 23)
	xscale = xl.xlnsv(24.0, 23)
	for iy in range(11,-12,-1):
		for ix in range(-40, 39):
			y = iy/yscale
			x = ix/xscale
			x1 = x
			y1 = y
			count = 0
			while ((x*x+y*y < four) and (count<iter)):
				xnew = x*x - y*y + x1
				ynew = x*y*two + y1
				count += 1
				x = xnew
				y = ynew
			if (count < iter):
				print("*",end="")
			else:
				print(" ",end="")
		print("")


def test1xlns32_float():
	odd = xl.xlnsv(1, 23)
	sum = xl.xlnsv(0, 23)
	for i in range(1,10001):
		sum += odd
		odd += 2.0
	print("test1xlns_float odd="+str(odd)+" sum="+str(sum))


def test2xlns32_float():
	num = xl.xlnsv(1.0, 23)
	fact = xl.xlnsv(1.0, 23)
	sum = xl.xlnsv(0.0, 23)
	for i in range(1,9):
		sum = sum + 1.0/fact
		fact = fact * num
		num = num + 1.0
	print("test2xlns_float num="+str(num)+" fact="+str(fact)+" sum="+str(sum))

def test3xlns32_float():
	num = xl.xlnsv(1.0, 23)
	fact = xl.xlnsv(1.0, 23)
	sum = xl.xlnsv(0.0, 23)
	for i in range(1,11):
		sum = sum + 1.0/fact
		fact = -fact * num * (num + 1.0)
		num = num + 2.0
	print("test3xlns_float num="+str(num)+" fact="+str(fact)+" sum="+str(sum))

def main():
	print("xlns32 Python doing same tests as the C++ version (32-bit like float)");

	test5fp();
	test5xlns32_float();
	test1fp();
	test1xlns32_float();
	test2fp();
	test2xlns32_float();
	test3fp();
	test3xlns32_float();

	test4fp(2000);
	test4xlns32_float(2000);


main()

