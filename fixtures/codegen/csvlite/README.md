# csvlite.py

Implement a small CSV reader/writer. Pure standard library. Do not use the `csv` module.

`parse(text)` returns a list of rows, each a list of strings:
- `,` separates fields.
- A field starting with `"` is quoted. Inside a quoted field, `""` means a literal `"`, and
  commas and newlines are ordinary characters.
- `\n`, `\r\n` and a **missing final newline** all end a row.
- A blank line produces an empty row (a row with one empty field).
- Empty input returns `[]`.
- The result always has the same number of rows as the input has lines.

`format(rows)` turns rows back into text:
- Fields are quoted **only when necessary**: when the field contains `,`, `"`, `\r` or `\n`,
  or has leading or trailing whitespace. An empty field is never quoted.
- A literal `"` inside a quoted field is written `""`.
- Rows are joined with `\n`; there is **no trailing newline**.

`format(parse(t))` must reproduce `t` for canonical input.

Return **only** the complete file contents in a single ```python code block, nothing else.
