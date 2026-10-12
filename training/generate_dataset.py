
"""
CodeInsight AI - Dataset Generation Pipeline

Generates a labelled dataset of beginner Python programs with controlled mutations.
Each mutation introduces exactly one logical error category.
All generated programs are validated via AST parsing and tested execution.

Classes:
  - correct
  - wrong_initialization
  - off_by_one_loop
  - wrong_operator
  - wrong_condition
"""

import ast
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import textwrap
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Constants ───────────────────────────────────────────────────────────────

ERROR_LABELS = [
    "correct",
    "wrong_initialization",
    "off_by_one_loop",
    "wrong_operator",
    "wrong_condition",
]

SEED = 42

# ─── Template Registry ───────────────────────────────────────────────────────
# Each template: (template_id, problem_type, code, test_fn, expected_result)
# test_fn returns True if the function behaves correctly

TEMPLATES: List[Dict] = []


def register_template(template_id, problem_type, code, test_code):
    """Register a correct program template with its test."""
    TEMPLATES.append({
        "template_id": template_id,
        "problem_type": problem_type,
        "code": textwrap.dedent(code).strip(),
        "test_code": textwrap.dedent(test_code).strip(),
    })


# ─── Correct Program Templates ──────────────────────────────────────────────

register_template("factorial_1", "factorial", """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
""", """
assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(5) == 120
assert factorial(3) == 6
""")

register_template("factorial_2", "factorial", """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""", """
assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(5) == 120
""")

register_template("factorial_3", "factorial", """
def factorial(n):
    result = 1
    i = 2
    while i <= n:
        result *= i
        i += 1
    return result
""", """
assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(5) == 120
assert factorial(4) == 24
""")

register_template("fibonacci_1", "fibonacci", """
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a = 0
    b = 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
""", """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(5) == 5
assert fibonacci(7) == 13
""")

register_template("fibonacci_2", "fibonacci", """
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
""", """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(6) == 8
""")

register_template("is_prime_1", "prime", """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""", """
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(17) == True
assert is_prime(1) == False
assert is_prime(0) == False
""")

register_template("is_prime_2", "prime", """
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
""", """
assert is_prime(2) == True
assert is_prime(9) == False
assert is_prime(11) == True
assert is_prime(15) == False
""")

register_template("palindrome_1", "palindrome", """
def is_palindrome(s):
    left = 0
    right = len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
""", """
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
assert is_palindrome("aba") == True
assert is_palindrome("a") == True
assert is_palindrome("") == True
""")

register_template("palindrome_2", "palindrome", """
def is_palindrome(s):
    return s == s[::-1]
""", """
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
assert is_palindrome("abba") == True
""")

register_template("max_element_1", "max_element", """
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
    return max_val
""", """
assert find_max([3, 1, 4, 1, 5]) == 5
assert find_max([1]) == 1
assert find_max([-1, -5, -2]) == -1
assert find_max([]) is None
""")

register_template("max_element_2", "max_element", """
def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val
""", """
assert find_max([10, 20, 5]) == 20
assert find_max([-3]) == -3
assert find_max([]) is None
""")

register_template("sum_list_1", "sum_list", """
def sum_list(lst):
    total = 0
    for item in lst:
        total += item
    return total
""", """
assert sum_list([1, 2, 3]) == 6
assert sum_list([]) == 0
assert sum_list([10]) == 10
assert sum_list([-1, 1]) == 0
""")

register_template("sum_list_2", "sum_list", """
def sum_list(lst):
    total = 0
    for i in range(len(lst)):
        total = total + lst[i]
    return total
""", """
assert sum_list([1, 2, 3, 4]) == 10
assert sum_list([]) == 0
assert sum_list([5]) == 5
""")

register_template("linear_search_1", "linear_search", """
def linear_search(lst, target):
    for i in range(len(lst)):
        if lst[i] == target:
            return i
    return -1
""", """
assert linear_search([1, 2, 3, 4], 3) == 2
assert linear_search([1, 2, 3], 5) == -1
assert linear_search([], 1) == -1
assert linear_search([10], 10) == 0
""")

register_template("linear_search_2", "linear_search", """
def linear_search(lst, target):
    index = 0
    while index < len(lst):
        if lst[index] == target:
            return index
        index += 1
    return -1
""", """
assert linear_search([5, 10, 15], 10) == 1
assert linear_search([5, 10, 15], 20) == -1
""")

register_template("reverse_string_1", "reverse_string", """
def reverse_string(s):
    result = ""
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result
""", """
assert reverse_string("hello") == "olleh"
assert reverse_string("") == ""
assert reverse_string("a") == "a"
assert reverse_string("ab") == "ba"
""")

register_template("reverse_string_2", "reverse_string", """
def reverse_string(s):
    chars = list(s)
    left = 0
    right = len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return "".join(chars)
""", """
assert reverse_string("hello") == "olleh"
assert reverse_string("abc") == "cba"
assert reverse_string("") == ""
""")

register_template("bubble_sort_1", "sorting", """
def bubble_sort(lst):
    arr = lst[:]
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""", """
assert bubble_sort([3, 1, 2]) == [1, 2, 3]
assert bubble_sort([]) == []
assert bubble_sort([1]) == [1]
assert bubble_sort([5, 3, 8, 1]) == [1, 3, 5, 8]
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
assert count_vowels("") == 0
assert count_vowels("AEIOU") == 5
assert count_vowels("xyz") == 0
""")

register_template("power_1", "power", """
def power(base, exp):
    result = 1
    for i in range(exp):
        result *= base
    return result
""", """
assert power(2, 3) == 8
assert power(5, 0) == 1
assert power(3, 2) == 9
assert power(10, 1) == 10
""")

register_template("gcd_1", "gcd", """
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
""", """
assert gcd(12, 8) == 4
assert gcd(7, 3) == 1
assert gcd(10, 5) == 5
assert gcd(1, 1) == 1
""")


# ─── Mutation Engine ─────────────────────────────────────────────────────────

# Each mutation function takes (template) and returns list of
# (mutated_code, error_label, mutation_type, mutation_description)

def generate_wrong_initialization_mutations(template: Dict) -> List[Dict]:
    """Generate wrong_initialization mutations."""
    mutations = []
    code = template["code"]
    
    # Pattern: result = 1 -> result = 0 (for multiplication accumulator)
    if "result = 1" in code and ("result *" in code or "result = result *" in code or "result *=" in code):
        mutated = code.replace("result = 1", "result = 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "accumulator_init",
            "mutation_description": "Changed multiplication accumulator from 1 to 0",
        })
    
    # Pattern: total = 0 -> total = 1 (for addition accumulator)
    if "total = 0" in code and ("total +=" in code or "total = total +" in code):
        mutated = code.replace("total = 0", "total = 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "accumulator_init",
            "mutation_description": "Changed addition accumulator from 0 to 1",
        })
    
    # Pattern: count = 0 -> count = 1
    if "count = 0" in code and "count +=" in code:
        mutated = code.replace("count = 0", "count = 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "counter_init",
            "mutation_description": "Changed counter initial value from 0 to 1",
        })
    
    # Pattern: max_val = lst[0] -> max_val = 0
    if "max_val = lst[0]" in code:
        mutated = code.replace("max_val = lst[0]", "max_val = 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "variable_init",
            "mutation_description": "Changed max initialization from lst[0] to 0 (fails for negative lists)",
        })
    
    # Pattern: a = 0 / b = 1 in fibonacci -> a = 1
    if "a = 0\n" in code and "b = 1" in code:
        mutated = code.replace("a = 0\n", "a = 1\n", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "variable_init",
            "mutation_description": "Changed fibonacci initial value a from 0 to 1",
        })
    
    # Pattern: left = 0 -> left = 1
    if "left = 0" in code and "left < right" in code:
        mutated = code.replace("left = 0", "left = 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "pointer_init",
            "mutation_description": "Changed left pointer initial value from 0 to 1",
        })
    
    # Pattern: index = 0 -> index = 1
    if "index = 0" in code and "index < len" in code:
        mutated = code.replace("index = 0", "index = 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "index_init",
            "mutation_description": "Changed starting index from 0 to 1",
        })
    
    # Pattern: result = "" -> result = " "
    if 'result = ""' in code:
        mutated = code.replace('result = ""', 'result = " "', 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "string_init",
            "mutation_description": "Changed empty string initialization to space character",
        })
    
    # Pattern: i = 2 -> i = 1 (for primes starting check)
    if "i = 2" in code and "i * i" not in code:
        pass  # skip ambiguous cases
    elif "i = 2" in code and "i * i" in code:
        mutated = code.replace("i = 2", "i = 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "loop_var_init",
            "mutation_description": "Changed prime check starting value from 2 to 1",
        })
    elif "i = 2" in code:
        mutated = code.replace("i = 2", "i = 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_initialization",
            "mutation_type": "loop_var_init",
            "mutation_description": "Changed loop start variable from 2 to 0",
        })
    
    return mutations


def generate_off_by_one_mutations(template: Dict) -> List[Dict]:
    """Generate off_by_one_loop mutations."""
    mutations = []
    code = template["code"]
    
    # Pattern: range(1, n + 1) -> range(1, n) (missing last element)
    if "range(1, n + 1)" in code:
        mutated = code.replace("range(1, n + 1)", "range(1, n)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_endpoint",
            "mutation_description": "Changed range(1, n+1) to range(1, n) - misses last iteration",
        })
    
    # Pattern: range(2, n + 1) -> range(2, n)
    if "range(2, n + 1)" in code:
        mutated = code.replace("range(2, n + 1)", "range(2, n)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_endpoint",
            "mutation_description": "Changed range(2, n+1) to range(2, n) - misses last iteration",
        })
    
    # Pattern: range(len(s) - 1, -1, -1) -> range(len(s) - 2, -1, -1)
    if "range(len(s) - 1, -1, -1)" in code:
        mutated = code.replace("range(len(s) - 1, -1, -1)", "range(len(s) - 2, -1, -1)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_start",
            "mutation_description": "Changed reverse loop start, skipping last character",
        })
    
    # Pattern: range(len(lst)) -> range(len(lst) - 1)
    if "range(len(lst))" in code and "range(len(lst) - 1)" not in code:
        mutated = code.replace("range(len(lst))", "range(len(lst) - 1)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_endpoint",
            "mutation_description": "Shortened range by 1, missing last element",
        })
    
    # Pattern: range(1, len(lst)) -> range(1, len(lst) - 1)
    if "range(1, len(lst))" in code:
        mutated = code.replace("range(1, len(lst))", "range(1, len(lst) - 1)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_endpoint",
            "mutation_description": "Shortened range by 1, missing last comparison",
        })
    
    # Pattern: while i <= n -> while i < n
    if "while i <= n:" in code:
        mutated = code.replace("while i <= n:", "while i < n:", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "loop_boundary",
            "mutation_description": "Changed while i <= n to while i < n, missing last iteration",
        })
    
    # Pattern: while left < right -> while left <= right (off-by-one for palindrome)
    # This actually causes an issue for even-length strings
    if "while left < right:" in code and "palindrome" in template.get("problem_type", ""):
        mutated = code.replace("while left < right:", "while left <= right:", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "loop_boundary",
            "mutation_description": "Changed < to <= in pointer comparison, causes index issues",
        })
    
    # Pattern: range(0, n - i - 1) -> range(0, n - i)
    if "range(0, n - i - 1)" in code:
        mutated = code.replace("range(0, n - i - 1)", "range(0, n - i)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_endpoint",
            "mutation_description": "Changed bubble sort inner loop boundary, causes index out of range",
        })
    
    # Pattern: range(n) -> range(n - 1) for bubble sort outer
    if "range(n)" in code and "bubble" in template.get("template_id", ""):
        mutated = code.replace("for i in range(n):", "for i in range(n - 1):", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_endpoint",
            "mutation_description": "Reduced bubble sort passes by 1",
        })
    
    # Pattern: while index < len(lst) -> while index < len(lst) - 1
    if "while index < len(lst):" in code:
        mutated = code.replace("while index < len(lst):", "while index < len(lst) - 1:", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "loop_boundary",
            "mutation_description": "Changed loop boundary to skip last element",
        })
    
    # Pattern: int(n ** 0.5) + 1 -> int(n ** 0.5)
    if "int(n ** 0.5) + 1" in code:
        mutated = code.replace("int(n ** 0.5) + 1", "int(n ** 0.5)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "off_by_one_loop",
            "mutation_type": "range_endpoint",
            "mutation_description": "Removed +1 from sqrt boundary, misses perfect square divisors",
        })
    
    return mutations


def generate_wrong_operator_mutations(template: Dict) -> List[Dict]:
    """Generate wrong_operator mutations."""
    mutations = []
    code = template["code"]
    
    # Pattern: result * i -> result + i
    if "result = result * i" in code:
        mutated = code.replace("result = result * i", "result = result + i", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed multiplication to addition in accumulator",
        })
    
    if "result *= i" in code:
        mutated = code.replace("result *= i", "result += i", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed *= to += in accumulator",
        })
    
    if "result *= base" in code:
        mutated = code.replace("result *= base", "result += base", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed *= to += in power calculation",
        })
    
    # Pattern: total += item -> total -= item
    if "total += item" in code:
        mutated = code.replace("total += item", "total -= item", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed += to -= in sum accumulator",
        })
    
    if "total = total + lst[i]" in code:
        mutated = code.replace("total = total + lst[i]", "total = total - lst[i]", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed + to - in sum accumulator",
        })
    
    # Pattern: n % i == 0 -> n // i == 0
    if "n % i == 0" in code:
        mutated = code.replace("n % i == 0", "n // i == 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed modulo (%) to integer division (//) in divisibility check",
        })
    
    # Pattern: n * factorial(n-1) -> n + factorial(n-1)
    if "n * factorial(n - 1)" in code:
        mutated = code.replace("n * factorial(n - 1)", "n + factorial(n - 1)", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed * to + in recursive factorial",
        })
    
    # Pattern: a + b in fibonacci -> a - b
    if "a + b" in code and "fibonacci" in template.get("problem_type", ""):
        mutated = code.replace("a + b", "a - b", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed + to - in Fibonacci recurrence",
        })
    
    # Pattern: lst[i] > max_val -> lst[i] < max_val
    if "lst[i] > max_val" in code:
        mutated = code.replace("lst[i] > max_val", "lst[i] < max_val", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "comparison_op",
            "mutation_description": "Changed > to < in max comparison (finds minimum instead)",
        })
    
    if "item > max_val" in code:
        mutated = code.replace("item > max_val", "item < max_val", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "comparison_op",
            "mutation_description": "Changed > to < in max comparison (finds minimum instead)",
        })
    
    # Pattern: arr[j] > arr[j + 1] -> arr[j] < arr[j + 1] (descending sort)
    if "arr[j] > arr[j + 1]" in code:
        mutated = code.replace("arr[j] > arr[j + 1]", "arr[j] < arr[j + 1]", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "comparison_op",
            "mutation_description": "Changed > to < in bubble sort comparison (sorts descending)",
        })
    
    # Pattern: left += 1 -> left -= 1
    if "left += 1" in code and "right -= 1" in code:
        mutated = code.replace("left += 1", "left -= 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "increment_op",
            "mutation_description": "Changed left pointer increment to decrement",
        })
    
    # Pattern: a % b -> a // b in GCD
    if "a % b" in code and "gcd" in template.get("problem_type", ""):
        mutated = code.replace("a % b", "a // b", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed modulo to integer division in GCD",
        })
    
    # Pattern: count += 1 -> count -= 1
    if "count += 1" in code:
        mutated = code.replace("count += 1", "count -= 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "arithmetic_op",
            "mutation_description": "Changed counter increment to decrement",
        })
    
    # Pattern: index += 1 -> index += 2
    if "index += 1" in code and "while index" in code:
        mutated = code.replace("index += 1", "index += 2", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_operator",
            "mutation_type": "step_op",
            "mutation_description": "Changed step size from 1 to 2, skipping elements",
        })
    
    return mutations


def generate_wrong_condition_mutations(template: Dict) -> List[Dict]:
    """Generate wrong_condition mutations."""
    mutations = []
    code = template["code"]
    
    # Pattern: n < 2 -> n < 1 in prime check
    if "n < 2" in code and "prime" in template.get("problem_type", ""):
        mutated = code.replace("n < 2", "n < 1", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "boundary_condition",
            "mutation_description": "Changed n < 2 to n < 1, incorrectly treating 1 as prime",
        })
    
    # Pattern: n <= 1 -> n <= 0 in factorial base case
    if "n <= 1" in code and "factorial" in template.get("problem_type", ""):
        mutated = code.replace("n <= 1", "n <= 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "base_case",
            "mutation_description": "Changed base case from n<=1 to n<=0, causing extra recursion",
        })
    
    # Pattern: n <= 0 -> n < 0 in fibonacci
    if "n <= 0" in code and "fibonacci" in template.get("problem_type", ""):
        mutated = code.replace("n <= 0", "n < 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "boundary_condition",
            "mutation_description": "Changed n <= 0 to n < 0, fails for fib(0)",
        })
    
    # Pattern: s[left] != s[right] -> s[left] == s[right]
    if "s[left] != s[right]" in code:
        mutated = code.replace("s[left] != s[right]", "s[left] == s[right]", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "logic_inversion",
            "mutation_description": "Inverted palindrome comparison from != to ==",
        })
    
    # Pattern: if not lst -> if lst
    if "if not lst:" in code:
        mutated = code.replace("if not lst:", "if lst:", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "logic_inversion",
            "mutation_description": "Inverted empty list check from 'not lst' to 'lst'",
        })
    
    # Pattern: lst[i] == target -> lst[i] != target
    if "lst[i] == target" in code:
        mutated = code.replace("lst[i] == target", "lst[i] != target", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "logic_inversion",
            "mutation_description": "Inverted search comparison from == to !=",
        })
    
    if "lst[index] == target" in code:
        mutated = code.replace("lst[index] == target", "lst[index] != target", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "logic_inversion",
            "mutation_description": "Inverted search comparison from == to !=",
        })
    
    # Pattern: n % 2 == 0 -> n % 2 != 0 (even check)
    if "n % 2 == 0" in code and "prime" in template.get("problem_type", ""):
        mutated = code.replace("n % 2 == 0", "n % 2 != 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "logic_inversion",
            "mutation_description": "Inverted even number check, skips odd numbers instead of even",
        })
    
    # Pattern: b != 0 -> b == 0 in GCD
    if "b != 0" in code and "gcd" in template.get("problem_type", ""):
        mutated = code.replace("b != 0", "b == 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "logic_inversion",
            "mutation_description": "Inverted GCD loop condition from != to ==",
        })
    
    # Pattern: char.lower() in "aeiou" -> char.lower() not in "aeiou"
    if 'char.lower() in "aeiou"' in code:
        mutated = code.replace('char.lower() in "aeiou"', 'char.lower() not in "aeiou"', 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "logic_inversion",
            "mutation_description": "Inverted vowel check, counts consonants instead",
        })
    
    # Pattern: n == 1 -> n == 0 in fibonacci
    if "n == 1" in code and "fibonacci" in template.get("problem_type", ""):
        mutated = code.replace("n == 1", "n == 0", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "boundary_condition",
            "mutation_description": "Changed fibonacci base case from n==1 to n==0",
        })
    
    # Pattern: n == 2 -> n == 3 in prime
    if "n == 2" in code and "prime" in template.get("problem_type", ""):
        mutated = code.replace("n == 2", "n == 3", 1)
        mutations.append({
            "code": mutated,
            "error_label": "wrong_condition",
            "mutation_type": "boundary_condition",
            "mutation_description": "Changed special case from n==2 to n==3",
        })
    
    return mutations


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_syntax(code: str) -> bool:
    """Check if code parses as valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def run_test_safely(code: str, test_code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Run test code in an isolated subprocess.
    Only used for trusted generated code, never for user submissions.
    
    Returns (passed, error_message).
    """
    full_code = code + "\n\n" + test_code
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr.strip().split("\n")[-1] if result.stderr else "Test failed"
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, str(e)


# ─── Dataset Assembly ────────────────────────────────────────────────────────

def generate_full_dataset(
    output_dir: Path,
    similarity_dir: Optional[Path] = None,
    seed: int = SEED,
) -> Dict:
    """
    Generate the complete labelled dataset.
    
    Returns summary dict with total_samples and class_distribution.
    """
    random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    rejected = []
    sample_counter = 0
    
    # Add correct programs
    for template in TEMPLATES:
        # Validate the correct template
        if not validate_syntax(template["code"]):
            rejected.append({
                "template_id": template["template_id"],
                "reason": "Template has syntax error",
            })
            continue
        
        passed, error = run_test_safely(template["code"], template["test_code"])
        if not passed:
            rejected.append({
                "template_id": template["template_id"],
                "reason": f"Template fails tests: {error}",
            })
            continue
        
        sample_counter += 1
        all_samples.append({
            "sample_id": f"S{sample_counter:04d}",
            "source_template_id": template["template_id"],
            "problem_type": template["problem_type"],
            "code": template["code"],
            "error_label": "correct",
            "mutation_type": "none",
            "mutation_description": "Original correct implementation",
            "expected_behavior": "passes_all_tests",
            "validation_status": "validated",
        })
    
    # Generate mutations
    mutation_generators = [
        generate_wrong_initialization_mutations,
        generate_off_by_one_mutations,
        generate_wrong_operator_mutations,
        generate_wrong_condition_mutations,
    ]
    
    for template in TEMPLATES:
        for gen_fn in mutation_generators:
            mutations = gen_fn(template)
            for mutation in mutations:
                mutated_code = mutation["code"]
                
                # Validate syntax
                if not validate_syntax(mutated_code):
                    rejected.append({
                        "template_id": template["template_id"],
                        "mutation_type": mutation["mutation_type"],
                        "reason": "Mutation introduced syntax error",
                    })
                    continue
                
                # Check that the mutation actually changes behavior
                passed, _ = run_test_safely(mutated_code, template["test_code"])
                if passed:
                    rejected.append({
                        "template_id": template["template_id"],
                        "mutation_type": mutation["mutation_type"],
                        "reason": "Mutation does not change test behavior - no observable error",
                    })
                    continue
                
                # Check for duplicate code
                code_hash = hashlib.md5(mutated_code.encode()).hexdigest()
                existing_hashes = {hashlib.md5(s["code"].encode()).hexdigest() for s in all_samples}
                if code_hash in existing_hashes:
                    rejected.append({
                        "template_id": template["template_id"],
                        "mutation_type": mutation["mutation_type"],
                        "reason": "Duplicate of existing sample",
                    })
                    continue
                
                sample_counter += 1
                all_samples.append({
                    "sample_id": f"S{sample_counter:04d}",
                    "source_template_id": template["template_id"],
                    "problem_type": template["problem_type"],
                    "code": mutated_code,
                    "error_label": mutation["error_label"],
                    "mutation_type": mutation["mutation_type"],
                    "mutation_description": mutation["mutation_description"],
                    "expected_behavior": "fails_tests",
                    "validation_status": "validated",
                })
    
    # Also generate some style variants for correct programs (augmentation)
    style_variants = _generate_style_variants(TEMPLATES)
    for variant in style_variants:
        sample_counter += 1
        variant["sample_id"] = f"S{sample_counter:04d}"
        all_samples.append(variant)
    
    # Create stratified splits (group by source_template_id)
    all_samples = _assign_splits(all_samples, seed=seed)
    
    # Write dataset
    csv_path = output_dir / "dataset.csv"
    fieldnames = [
        "sample_id", "source_template_id", "problem_type", "code",
        "error_label", "mutation_type", "mutation_description",
        "expected_behavior", "validation_status", "split",
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in all_samples:
            writer.writerow(sample)
    
    # Write rejected mutations log
    rejected_path = output_dir / "rejected_mutations.json"
    with open(rejected_path, "w") as f:
        json.dump(rejected, f, indent=2)
    
    # Generate similarity pairs
    if similarity_dir:
        _generate_similarity_pairs(all_samples, similarity_dir, seed)
    
    # Summary
    from collections import Counter
    class_dist = Counter(s["error_label"] for s in all_samples)
    
    summary = {
        "total_samples": len(all_samples),
        "class_distribution": dict(class_dist),
        "rejected_count": len(rejected),
        "templates_used": len(TEMPLATES),
    }
    
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Dataset generated: {len(all_samples)} samples")
    print(f"   Class distribution: {dict(class_dist)}")
    print(f"   Rejected mutations: {len(rejected)}")
    print(f"   Saved to: {csv_path}")
    
    return summary


def _generate_style_variants(templates: List[Dict]) -> List[Dict]:
    """Generate stylistic variants of correct programs for augmentation."""
    variants = []
    
    for template in templates:
        code = template["code"]
        
        # Variant: add docstring
        lines = code.split("\n")
        if lines and lines[0].strip().startswith("def "):
            new_lines = [lines[0], '    """Compute the result."""'] + lines[1:]
            variant_code = "\n".join(new_lines)
            if validate_syntax(variant_code):
                passed, _ = run_test_safely(variant_code, template["test_code"])
                if passed:
                    variants.append({
                        "source_template_id": template["template_id"],
                        "problem_type": template["problem_type"],
                        "code": variant_code,
                        "error_label": "correct",
                        "mutation_type": "style_variant",
                        "mutation_description": "Added docstring to function",
                        "expected_behavior": "passes_all_tests",
                        "validation_status": "validated",
                    })
    
    return variants


def _assign_splits(samples: List[Dict], seed: int = 42) -> List[Dict]:
    """
    Assign train/val/test splits grouped by source_template_id.
    All variants from the same template stay in the same split.
    Approximate ratio: 70% train, 15% val, 15% test.
    """
    random.seed(seed)
    
    # Group by template
    template_groups = defaultdict(list)
    for i, sample in enumerate(samples):
        template_groups[sample["source_template_id"]].append(i)
    
    # Shuffle template IDs
    template_ids = list(template_groups.keys())
    random.shuffle(template_ids)
    
    n = len(template_ids)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    for i, tid in enumerate(template_ids):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"
        
        for idx in template_groups[tid]:
            samples[idx]["split"] = split
    
    return samples


def _generate_similarity_pairs(
    samples: List[Dict],
    output_dir: Path,
    seed: int = 42,
) -> None:
    """
    Generate labelled code similarity pairs from the classification dataset.
    
    Similar pairs: same problem_type, both correct (or same template)
    Dissimilar pairs: different problem_type
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    correct_by_problem = defaultdict(list)
    for s in samples:
        if s["error_label"] == "correct":
            correct_by_problem[s["problem_type"]].append(s)
    
    pairs = []
    pair_id = 0
    
    # Similar pairs: same problem type
    problem_types = list(correct_by_problem.keys())
    for pt in problem_types:
        group = correct_by_problem[pt]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pair_id += 1
                split_a = group[i].get("split", "train")
                split_b = group[j].get("split", "train")
                pairs.append({
                    "pair_id": f"P{pair_id:04d}",
                    "code_a": group[i]["code"],
                    "code_b": group[j]["code"],
                    "label": 1,  # similar
                    "source": f"{group[i]['source_template_id']}+{group[j]['source_template_id']}",
                    "split": split_a if split_a == split_b else "cross_split",
                })
    
    # Dissimilar pairs: different problem types
    for i in range(len(problem_types)):
        for j in range(i + 1, len(problem_types)):
            group_a = correct_by_problem[problem_types[i]]
            group_b = correct_by_problem[problem_types[j]]
            
            # Sample up to 2 pairs per problem-type pair
            combos = [(a, b) for a in group_a for b in group_b]
            random.shuffle(combos)
            for a, b in combos[:2]:
                pair_id += 1
                split_a = a.get("split", "train")
                split_b = b.get("split", "train")
                pairs.append({
                    "pair_id": f"P{pair_id:04d}",
                    "code_a": a["code"],
                    "code_b": b["code"],
                    "label": 0,  # dissimilar
                    "source": f"{a['source_template_id']}+{b['source_template_id']}",
                    "split": split_a if split_a == split_b else "cross_split",
                })
    
    # Write pairs
    csv_path = output_dir / "pairs.csv"
    fieldnames = ["pair_id", "code_a", "code_b", "label", "source", "split"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)
    
    print(f"   Similarity pairs: {len(pairs)} (saved to {csv_path})")


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    
    output_dir = project_root / "data" / "classification"
    similarity_dir = project_root / "data" / "similarity"
    
    print("=" * 60)
    print("CodeInsight AI - Dataset Generation")
    print("=" * 60)
    print(f"Templates: {len(TEMPLATES)}")
    print(f"Output: {output_dir}")
    print()
    
    result = generate_full_dataset(output_dir, similarity_dir)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(json.dumps(result, indent=2))
