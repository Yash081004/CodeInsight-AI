"""Additional program families for benchmark_v3 (unique problem_type per family)."""

from training.generate_dataset import register_template

# 32 new independent families toward 50+ total with v1/v2 templates.

register_template("binary_search_v3", "binary_search", """
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""", """
assert binary_search([1, 3, 5, 7], 5) == 2
assert binary_search([1, 3, 5, 7], 4) == -1
""")

register_template("insertion_sort_v3", "insertion_sort", """
def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a
""", """
assert insertion_sort([3, 1, 2]) == [1, 2, 3]
""")

register_template("merge_lists_v3", "merge_sorted_lists", """
def merge(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
""", """
assert merge([1, 3], [2, 4]) == [1, 2, 3, 4]
""")

register_template("count_chars_v3", "count_char", """
def count_char(s, ch):
    c = 0
    for x in s:
        if x == ch:
            c += 1
    return c
""", """
assert count_char("banana", "a") == 3
assert count_char("", "a") == 0
""")

register_template("strip_spaces_v3", "strip_spaces", """
def strip_spaces(s):
    return "".join(ch for ch in s if ch != " ")
""", """
assert strip_spaces("a b c") == "abc"
""")

register_template("last_index_v3", "last_index", """
def last_index(lst, value):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == value:
            return i
    return -1
""", """
assert last_index([1, 2, 3, 2], 2) == 3
""")

register_template("tuple_sum_v3", "tuple_sum", """
def tuple_sum(t):
    s = 0
    for x in t:
        s += x
    return s
""", """
assert tuple_sum((1, 2, 3)) == 6
""")

register_template("is_even_v3", "is_even", """
def is_even(n):
    return n % 2 == 0
""", """
assert is_even(4) == True
assert is_even(3) == False
""")

register_template("double_list_v3", "double_list", """
def double_list(lst):
    return [x * 2 for x in lst]
""", """
assert double_list([1, 2]) == [2, 4]
""")

register_template("list_max_idx_v3", "list_max_index", """
def max_index(lst):
    if not lst:
        return -1
    best_i = 0
    for i in range(1, len(lst)):
        if lst[i] > lst[best_i]:
            best_i = i
    return best_i
""", """
assert max_index([3, 1, 4]) == 2
""")

register_template("remove_neg_v3", "remove_negatives", """
def remove_negatives(lst):
    out = []
    for x in lst:
        if x >= 0:
            out.append(x)
    return out
""", """
assert remove_negatives([1, -2, 3]) == [1, 3]
""")

register_template("cumulative_sum_v3", "cumulative_sum", """
def cumulative_sum(lst):
    out = []
    t = 0
    for x in lst:
        t += x
        out.append(t)
    return out
""", """
assert cumulative_sum([1, 2, 3]) == [1, 3, 6]
""")

register_template("count_pos_v3", "count_positive", """
def count_positive(lst):
    c = 0
    for x in lst:
        if x > 0:
            c += 1
    return c
""", """
assert count_positive([0, 1, -1, 2]) == 2
""")

register_template("string_len_v3", "string_len", """
def string_len(s):
    n = 0
    for _ in s:
        n += 1
    return n
""", """
assert string_len("abc") == 3
""")

register_template("starts_with_v3", "starts_with", """
def starts_with(s, prefix):
    if len(prefix) > len(s):
        return False
    for i in range(len(prefix)):
        if s[i] != prefix[i]:
            return False
    return True
""", """
assert starts_with("hello", "he") == True
assert starts_with("hello", "x") == False
""")

register_template("ends_with_v3", "ends_with", """
def ends_with(s, suffix):
    if len(suffix) > len(s):
        return False
    start = len(s) - len(suffix)
    for i in range(len(suffix)):
        if s[start + i] != suffix[i]:
            return False
    return True
""", """
assert ends_with("hello", "lo") == True
""")

register_template("rotate_left_v3", "rotate_left", """
def rotate_left(lst, k):
    if not lst:
        return lst
    k = k % len(lst)
    return lst[k:] + lst[:k]
""", """
assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]
""")

register_template("unique_count_v3", "unique_count", """
def unique_count(lst):
    seen = set()
    for x in lst:
        seen.add(x)
    return len(seen)
""", """
assert unique_count([1, 1, 2]) == 2
""")

register_template("second_largest_v3", "second_largest", """
def second_largest(lst):
    if len(lst) < 2:
        return None
    a = max(lst)
    rest = [x for x in lst if x != a]
    if not rest:
        return None
    return max(rest)
""", """
assert second_largest([1, 3, 2, 3]) == 2
""")

register_template("digit_count_v3", "digit_count", """
def digit_count(n):
    n = abs(n)
    if n == 0:
        return 1
    c = 0
    while n:
        c += 1
        n //= 10
    return c
""", """
assert digit_count(102) == 3
assert digit_count(0) == 1
""")

register_template("is_ascending_v3", "is_ascending", """
def is_ascending(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True
""", """
assert is_ascending([1, 2, 2]) == True
""")

register_template("pair_sum_target_v3", "pair_sum_target", """
def has_pair_sum(arr, target):
    seen = set()
    for x in arr:
        if target - x in seen:
            return True
        seen.add(x)
    return False
""", """
assert has_pair_sum([1, 2, 3], 5) == True
assert has_pair_sum([1, 2], 10) == False
""")

register_template("matrix_trace_v3", "matrix_trace", """
def matrix_trace(m):
    t = 0
    for i in range(len(m)):
        t += m[i][i]
    return t
""", """
assert matrix_trace([[1, 2], [3, 4]]) == 5
""")

register_template("flatten_once_v3", "flatten_once", """
def flatten_once(nested):
    out = []
    for sub in nested:
        for x in sub:
            out.append(x)
    return out
""", """
assert flatten_once([[1, 2], [3]]) == [1, 2, 3]
""")

register_template("running_max_v3", "running_max", """
def running_max(lst):
    out = []
    cur = None
    for x in lst:
        if cur is None or x > cur:
            cur = x
        out.append(cur)
    return out
""", """
assert running_max([1, 3, 2]) == [1, 3, 3]
""")

register_template("count_words_v3", "count_words", """
def count_words(s):
    parts = s.split()
    return len(parts)
""", """
assert count_words("a bb ccc") == 3
assert count_words("") == 0
""")

register_template("replace_char_v3", "replace_char", """
def replace_char(s, old, new):
    out = []
    for ch in s:
        out.append(new if ch == old else ch)
    return "".join(out)
""", """
assert replace_char("aba", "a", "x") == "xbx"
""")

register_template("zip_sum_v3", "zip_sum", """
def zip_sum(a, b):
    n = min(len(a), len(b))
    return sum(a[i] + b[i] for i in range(n))
""", """
assert zip_sum([1, 2], [3, 4]) == 10
""")

register_template("mod_list_v3", "mod_list", """
def mod_list(lst, m):
    return [x % m for x in lst]
""", """
assert mod_list([5, 7, 8], 3) == [2, 1, 2]
""")

register_template("sign_list_v3", "sign_list", """
def sign_list(lst):
    out = []
    for x in lst:
        if x > 0:
            out.append(1)
        elif x < 0:
            out.append(-1)
        else:
            out.append(0)
    return out
""", """
assert sign_list([-1, 0, 2]) == [-1, 0, 1]
""")

register_template("copy_matrix_v3", "copy_matrix", """
def copy_matrix(m):
    return [row[:] for row in m]
""", """
a = [[1, 2], [3, 4]]
b = copy_matrix(a)
b[0][0] = 9
assert a[0][0] == 1
""")

register_template("col_sum_v3", "col_sum", """
def col_sum(m, j):
    s = 0
    for row in m:
        s += row[j]
    return s
""", """
assert col_sum([[1, 2], [3, 4]], 0) == 4
""")

register_template("is_pal_num_v3", "is_palindrome_num", """
def is_palindrome_num(n):
    s = str(n)
    return s == s[::-1]
""", """
assert is_palindrome_num(121) == True
assert is_palindrome_num(123) == False
""")
