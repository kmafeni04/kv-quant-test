# fixedpoint.py

Implement exact decimal arithmetic for money, using **scaled integers**. Pure standard library.

Fixed point with `SCALE = 10000` (4 decimal places). A value is stored as an integer equal to
the real value times 10000. For example `12.3456` is stored as `123456`.

| function | behaviour |
|---|---|
| `parse(text)` | decimal string to scaled integer. Accepts an optional leading `+`/`-`, an optional single `.`, and **at most 4** decimal places. Raises `ValueError` on anything else (empty string, non-digits, more than 4 decimals, more than one `.`). `".5"` and `"5."` are both valid. |
| `format(value)` | scaled integer back to a string with **exactly 4** decimal places, e.g. `123456 -> "12.3456"`, `-5 -> "-0.0005"` |
| `add(a, b)` / `sub(a, b)` | scaled-integer addition / subtraction |
| `mul(a, b)` | exact product, then rounded to the nearest scaled integer, **halves away from zero** |
| `div(a, b)` | exact quotient, then rounded the same way. Raises `ZeroDivisionError` when `b == 0` |
| `to_float(value)` | the value as a Python float |

The point of the module is exactness: `parse("0.1") + parse("0.2")` must equal `parse("0.3")`
**exactly**, with no floating-point error anywhere in `add`/`sub`/`mul`/`div`.

Return **only** the complete file contents in a single ```python code block, nothing else.
