# csvlite.py

Implement a small CSV reader and writer for this fixture. Use only the Python standard library, and
do not use the `csv` module: the point of the task is to handle quoting and row endings yourself.

`parse(text)` returns a list of rows, where each row is a list of field strings. These rules decide
where a field ends and where a row ends:

- A comma separates fields.
- A field that starts with `"` is quoted. Inside a quoted field, `""` stands for one literal `"`, and
  commas and newlines are ordinary characters rather than separators.
- `\n`, `\r\n` and the end of the input with no final newline all end a row. A missing final newline
  is in that list deliberately, because it is the case hand-written parsers usually drop.
- A blank line produces an empty row, which is a row holding one empty field.
- Empty input returns `[]`, not a row with one empty field.
- The result always has the same number of rows as the input has lines. The hidden checks use that
  invariant to tell an off-by-one row count apart from a quoting mistake.

`format(rows)` turns rows back into text:

- A field is quoted **only when necessary**: when it contains `,`, `"`, `\r` or `\n`, or when it has
  leading or trailing whitespace. An empty field is never quoted.
- A literal `"` inside a quoted field is written `""`.
- Rows are joined with `\n` and there is **no trailing newline**, so the text ends at the last
  character of the last field.

`format(parse(t))` must reproduce `t` for canonical input, meaning input that already uses `\n` and
quotes only where the rules above require it.

Return **only** the complete file contents in a single ```python code block, nothing else.
