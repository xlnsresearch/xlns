import xlns as xl 
import numpy as np
import math
import timeit

def basel_sum(valuelist: list):
    valuelist = 1/(valuelist * valuelist)
    return valuelist.sum()  #np.sum does not work with xlnsnp

def run(terms: float):
    i = np.arange(1,terms)
    li = np.array(xl.xlnscopy(i))
    xi = xl.xlnsnp(i)
    print(terms, "\t", end='')

    # floats
    start = timeit.default_timer()
    fpep = (6 * basel_sum(i))**.5
    print(f" FP err: {math.pi - fpep}\t", end='')
    #print(" time: ", timeit.default_timer() - start, end='')

    # xlns
    start = timeit.default_timer()
    xlep = (6 * basel_sum(li))**.5
    print(f"xlns err: {math.pi - xlep}\t", end='')
    #print(" time: ", timeit.default_timer() - start)

    # xlnsnp
    start = timeit.default_timer()
    xlep = (6 * basel_sum(xi))**.5
    print(f"xlnsnp e: {math.pi - xlep}\t")   #, end='')
    #print(" time: ", timeit.default_timer() - start)

def main():
    for k in range(2,6):
        run(float(10**k))

if __name__ == "__main__":
    main()
