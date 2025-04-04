"""
Benchmark and validate a proposed safer __truediv__ implementation for xlnsnp,
handling division by zero (x / 0) by returning ±infinity depending on sign,
while maintaining original behavior for all other divisions.
"""

import numpy as np
import time
import xlns as xl


original_div = xl.xlnsnp.__truediv__

def new_div(self, v):
    t = xl.xlnsnp("")
    INT64_MIN = np.int64(-9223372036854775808)  # special zero
    INT64_MAX_POS = np.int64(9223372036854775806)  # +inf
    INT64_MAX_NEG = np.int64(9223372036854775807)  # -inf. INT64 max
 


    is_inf = np.logical_and(self.nd != INT64_MIN, v.nd == INT64_MIN)

    normal_div = (self.nd - v.nd + (v.nd & 1)) ^ (v.nd & 1)
    inf_result = np.where(self.nd & 1 == 1, INT64_MAX_NEG, INT64_MAX_POS)

    t.nd = np.where(is_inf, inf_result,
             np.where(self.nd == INT64_MIN, INT64_MIN, normal_div))
    return t


N = 10_000_000
np.random.seed(42)

rand_vals = np.random.randint(1, np.iinfo(np.int64).max, size=N, dtype=np.int64) | 1 #INT64_MAX_POS gets excluded for being even


x = xl.xlnsnp("")
x.nd = rand_vals.copy()


y_zero = xl.xlnsnp("")
y_zero.nd = np.full(N, -9223372036854775808, dtype=np.int64)


y_mixed = xl.xlnsnp("")
y_mixed.nd = np.where(np.random.rand(N) > 0.5, -9223372036854775808, rand_vals) 


y_nonzero = xl.xlnsnp("")
y_nonzero.nd = rand_vals.copy()

def run_and_time(name, x, y):
    start = time.time()
    result = x / y
    duration = time.time() - start
    print(f"{name:20s}: {duration:.4f} seconds")
    return result

# Benchmark Original
print("Running ORIGINAL __truediv__")
xl.xlnsnp.__truediv__ = original_div
res_orig_zero = run_and_time("orig: x / 0", x, y_zero)
res_orig_mix  = run_and_time("orig: x / mixed", x, y_mixed)
res_orig_nonz = run_and_time("orig: x / non-zero", x, y_nonzero)

#  Benchmark NEW 
print("\nRunning NEW __truediv__")
xl.xlnsnp.__truediv__ = new_div
res_new_zero = run_and_time("new: x / 0", x, y_zero)
res_new_mix  = run_and_time("new: x / mixed", x, y_mixed)
res_new_nonz = run_and_time("new: x / non-zero", x, y_nonzero)

# Checking correctness of the new truediv function
def compare_results(name, old, new):
    mismatch = np.count_nonzero(old.nd != new.nd)
    print(f"{name:20s} - Differences: {mismatch:,} / {N}")

print("\nComparing outputs:")
compare_results("x / 0", res_orig_zero, res_new_zero)
compare_results("x / mixed", res_orig_mix, res_new_mix)
compare_results("x / non-zero", res_orig_nonz, res_new_nonz)

#Ensuring correct inheritance of zero division behaviour in xlnsnpv subclass
print("\nChecking xlnsnpv subclass behavior")

x_v = xl.xlnsnpv(x)
y_v = xl.xlnsnpv(y_zero)


result_v = x_v / y_v


expected_v = xl.xlnsnpv(new_div(x, y_zero))


matches = np.array_equal(result_v.nd, expected_v.nd)
print("xlnsnpv output matches updated xlnsnp output:", matches)



# Restoring original __truediv__
xl.xlnsnp.__truediv__ = original_div
