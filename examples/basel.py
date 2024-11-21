import xlns as xl 
import numpy as np
import math
import timeit

def basel_sum(valuelist: list):
    valuelist = 1/(valuelist * valuelist)
    return np.sum(valuelist)

def run(terms: float):
    i = np.arange(1,terms)
    li = np.array(xl.xlnscopy(i))
    print(terms, "\t", end='')

    # floats
    start = timeit.default_timer()
    fpep = math.sqrt(6 * basel_sum(i))
    print(f" FP err: {math.pi - fpep}\t", end='')
    print(" time: ", timeit.default_timer() - start, end='')

    # xlns
    start = timeit.default_timer()
    xlep = math.sqrt(6 * basel_sum(li))
    print(f" LNS err: {math.pi - xlep}\t", end='')
    print(" time: ", timeit.default_timer() - start)

def main():
    for k in range(2,6):
        run(float(10**k))

if __name__ == "__main__":
    main()
