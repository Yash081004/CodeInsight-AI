"""
Additional program families for benchmark_v2 (registered into generate_dataset.TEMPLATES).
"""

from training.generate_dataset import register_template

# ─── New families (distinct problem_type = split group key in v2) ─────────────

register_template("gcd_iter", "gcd", """
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
""", """
assert gcd(12, 8) == 4
assert gcd(7, 3) == 1
assert gcd(0, 5) == 5
""")

register_template("reverse_list_1", "reverse_list", """
def reverse_list(lst):
    out = []
    for i in range(len(lst) - 1, -1, -1):
        out.append(lst[i])
    return out
""", """
assert reverse_list([1, 2, 3]) == [3, 2, 1]
assert reverse_list([]) == []
""")

register_template("count_vowels_1", "count_vowels", """
def count_vowels(s):
    count = 0
    for char in s:
        if char.lower() in "aeiou":
            count += 1
    return count
""", """
assert count_vowels("hello") == 2
assert count_vowels("xyz") == 0
""")

register_template("sum_digits_1", "sum_digits", """
def sum_digits(n):
    total = 0
    n = abs(n)
    while n > 0:
        total += n % 10
        n //= 10
    return total
""", """
assert sum_digits(123) == 6
assert sum_digits(0) == 0
""")

register_template("is_sorted_1", "is_sorted", """
def is_sorted(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True
""", """
assert is_sorted([1, 2, 3]) == True
assert is_sorted([3, 2, 1]) == False
assert is_sorted([]) == True
""")

register_template("find_min_1", "find_min", """
def find_min(lst):
    if not lst:
        return None
    m = lst[0]
    for x in lst[1:]:
        if x < m:
            m = x
    return m
""", """
assert find_min([3, 1, 4]) == 1
assert find_min([5]) == 5
""")

register_template("power_iter", "power", """
def power(base, exp):
    result = 1
    for _ in range(exp):
        result *= base
    return result
""", """
assert power(2, 3) == 8
assert power(5, 0) == 1
""")

register_template("linear_search_1", "linear_search", """
def linear_search(lst, target):
    for i in range(len(lst)):
        if lst[i] == target:
            return i
    return -1
""", """
assert linear_search([2, 4, 6], 4) == 1
assert linear_search([1], 2) == -1
""")

register_template("stack_push_pop", "stack_ops", """
def stack_height(ops):
    height = 0
    for op in ops:
        if op == "push":
            height += 1
        elif op == "pop" and height > 0:
            height -= 1
    return height
""", """
assert stack_height(["push", "push", "pop"]) == 1
assert stack_height(["pop"]) == 0
""")

register_template("mean_list_1", "mean_list", """
def mean_list(lst):
    if not lst:
        return 0
    return sum(lst) / len(lst)
""", """
assert mean_list([2, 4]) == 3.0
assert mean_list([5]) == 5.0
""")

register_template("count_evens_1", "count_evens", """
def count_evens(nums):
    c = 0
    for n in nums:
        if n % 2 == 0:
            c += 1
    return c
""", """
assert count_evens([1, 2, 3, 4]) == 2
assert count_evens([1, 3]) == 0
""")

register_template("clamp_1", "clamp", """
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
""", """
assert clamp(5, 0, 10) == 5
assert clamp(-1, 0, 10) == 0
assert clamp(11, 0, 10) == 10
""")
