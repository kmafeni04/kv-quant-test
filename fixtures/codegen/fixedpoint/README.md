# fixedpoint.py

Implement exact decimal arithmetic for money, using **scaled integers** and the Python standard
library only. Nothing in the arithmetic may pass through a float; floats are only used where a value
is explicitly converted at the end.

The scale is fixed at `SCALE = 10000`, which gives four decimal places. A value is stored as the
integer equal to the real value times 10000, so `12.3456` is stored as `123456`, and one cent is
stored as `100`.

- `parse(text)` converts a decimal string to a scaled integer. It accepts an optional leading `+` or
  `-`, at most one `.`, and at most four decimal places, so `".5"` and `"5."` are both valid. Anything
  else raises `ValueError`: an empty string, a non-digit character, a fifth decimal place, or a second
  `.`.
- `format(value)` converts a scaled integer back to a string with exactly four decimal places, so
  `123456` becomes `"12.3456"` and `-5` becomes `"-0.0005"`.
- `add(a, b)` and `sub(a, b)` add and subtract scaled integers directly, which is why the scale has to
  be the same on both sides.
- `mul(a, b)` multiplies the two real values exactly and then rounds to the nearest scaled integer,
  with halves rounded away from zero.
- `div(a, b)` divides exactly and rounds the same way. It raises `ZeroDivisionError` when `b == 0`.
- `to_float(value)` returns the value as a Python float for callers that need one. It is the only
  function allowed to touch a float.

The point of the module is exactness: `parse("0.1") + parse("0.2")` must equal `parse("0.3")`
**exactly**, with no floating-point error anywhere in `add`, `sub`, `mul` or `div`.

Return **only** the complete file contents in a single ```python code block, nothing else.
