import re
import unicodedata


# copied from datatransformation.py so live input is cleaned exactly like the training data
def clean_text(t):
    t = unicodedata.normalize("NFKC", str(t))            # unify odd unicode forms (ligatures, full-width chars)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)     # invisible zero-width characters
    t = t.replace("\u00a0", " ")                         # non-breaking space -> normal space
    t = t.replace("\r\n", "\n").replace("\r", "\n")      # unify line endings
    t = t.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}))  # curly -> straight quotes
    t = re.sub(r"[ \t]+", " ", t)                        # runs of spaces/tabs -> one space
    t = re.sub(r" *\n *", "\n", t)                       # trailing/leading spaces around line breaks
    t = re.sub(r"\n{3,}", "\n\n", t)                     # 3+ blank lines -> one blank line
    return t.strip()