import re
import tokenize
from io import StringIO


def contains_non_english(text):
    """Check if the text contains non-English characters (e.g., Korean, Chinese)."""
    return bool(re.search(r"[^\x00-\x7F]", text))


def extract_symbols(text):
    """Extract code symbols (e.g., variables, functions) from the text."""
    # Match common programming symbols like variable names, function names, etc.
    matches = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    return matches


def mask_symbols(text, symbols):
    """Replace symbols in the text with unique placeholders."""
    masked_text = text
    symbol_map = {}
    for i, symbol in enumerate(symbols):
        placeholder = f"__SYMBOL{i}__"
        symbol_map[placeholder] = symbol
        masked_text = masked_text.replace(symbol, placeholder)
    return masked_text, symbol_map


def unmask_symbols(text, symbol_map):
    """Replace placeholders in the text with original symbols."""
    for placeholder, symbol in symbol_map.items():
        text = text.replace(placeholder, symbol)
    return text


def transltate_comments(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        code = file.read()

    # Tokenize the file
    updated_code = []
    tokens = tokenize.generate_tokens(StringIO(code).readline)
    for token in tokens:
        tok_type, tok_string, start, end, line = token
        # If it's a comment
        if tok_type == tokenize.COMMENT:
            print(f"Found comment: {tok_string.strip()}")
            translated = input("Enter the translated comment: ")
            # Preserve the original leading whitespace
            start_line, start_col = start
            end_line, end_col = end
            print(f"Previous {tok_string=}(len=({len(tok_string)})), {start=}, {end=}")
            new_len = len(translated)
            new_end_col = start_col + new_len
            new_end = (end_line, new_end_col)
            updated_code.append((tok_type, f"# {translated}", start, new_end, line))
        else:
            updated_code.append((tok_type, tok_string, start, end, line))

    # Reconstruct the code
    new_code = tokenize.untokenize(updated_code)

    # Write back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(new_code)


if __name__ == "__main__":
    transltate_comments("test_trans.py")
