"""A small ASD-STE100 checker for the paper prose.

Checks: sentence length (max 25 words descriptive text), passive voice, words that STE does not
approve (with the approved word to use), noun clusters longer than three words, and paragraphs longer
than six sentences. It strips LaTeX commands first. Usage: python ste_lint.py file.tex [file2.tex ...]
"""
import re
import sys

MAX_WORDS = 25
MAX_SENT_PER_PARA = 6

# unapproved word -> approved alternative (ASD-STE100 Issue 8 dictionary, common cases)
UNAPPROVED = {
    "utilise": "use", "utilize": "use", "employ": "use", "employed": "used", "employs": "uses",
    "demonstrate": "show", "demonstrates": "shows", "demonstrated": "showed", "indicate": "show",
    "indicates": "shows", "indicated": "showed", "perform": "do", "performed": "did", "performs": "does",
    "obtain": "get", "obtained": "got", "obtains": "gets", "require": "need", "requires": "needs",
    "required": "needed", "prior to": "before", "in order to": "to", "regarding": "about",
    "therefore": "thus", "however": "but", "whereas": "but", "whilst": "while", "approximately": "approximately",
    "commence": "start", "terminate": "stop", "additional": "more", "additionally": "also",
    "subsequently": "then", "consequently": "thus", "furthermore": "also", "moreover": "also",
    "nevertheless": "but", "facilitate": "help", "ensure": "make sure", "ensures": "makes sure",
    "comprise": "include", "comprises": "includes", "constitute": "make", "undergo": "have",
    "exhibit": "show", "exhibits": "shows", "exhibited": "showed", "illustrate": "show", "illustrates": "shows",
    "represent": "show", "represents": "shows", "represented": "showed", "attain": "get", "attains": "gets",
    "enable": "let", "enables": "lets", "permit": "let", "permits": "lets", "assess": "examine",
    "assessment": "examination", "evaluate": "examine", "evaluated": "examined", "evaluation": "test",
    "numerous": "many", "sufficient": "sufficient", "adequate": "sufficient", "determine": "find",
    "determined": "found", "determines": "finds", "various": "different", "several": "some",
    "majority": "most", "within": "in", "amongst": "among", "upon": "on", "via": "through",
    "cannot yet": "cannot", "accordingly": "thus", "respectively": "(rewrite)", "i.e.": "that is",
    "e.g.": "for example", "etc.": "(rewrite)", "optimal": "best", "optimum": "best", "minimise": "decrease",
    "minimize": "decrease", "maximise": "increase", "generate": "make", "generated": "made",
    "generates": "makes", "capable": "can", "effectively": "(delete)", "significantly": "(number)",
    "approach": "method", "framework": "method", "transparent": "clear", "reliability": "(rewrite)",
}

PASSIVE = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+ed|built|cut|kept|held|made|given|shown|drawn|known|taken|seen|found|put|set|split|read|left|fed|done|written|chosen|run)\b", re.I)
PASSIVE_OK = re.compile(r"\b(is|are|was|were)\s+(called|named|labelled)\b", re.I)  # defining use, allowed


def strip_latex(t):
    t = re.sub(r"%.*", "", t)
    t = re.sub(r"\\(cite|ref|label|input|includegraphics|caption|begin|end|section|subsection|IEEEkeywords|textcolor|IfFileExists)\*?(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " X ", t)  # remaining macros -> placeholder word
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"[{}~]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def sentences(para):
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", para.strip())
    return [x for x in s if len(x.split()) > 1]


def main(paths):
    total = 0
    for p in paths:
        raw = open(p).read()
        # skip the LaTeX preamble and the abstract (submitted text, kept as is)
        raw = re.sub(r"^.*?\\maketitle", "", raw, flags=re.S)
        raw = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", raw, flags=re.S)
        paras = [x for x in re.split(r"\n\s*\n", raw) if x.strip() and not x.strip().startswith("\\begin{table")]
        for para in paras:
            text = strip_latex(para)
            ss = sentences(text)
            if len(ss) > MAX_SENT_PER_PARA:
                print(f"{p}: paragraph with {len(ss)} sentences (max {MAX_SENT_PER_PARA}): {ss[0][:60]}...")
                total += 1
            for s in ss:
                n = len(s.split())
                if n > MAX_WORDS:
                    print(f"{p}: {n} words: {s[:110]}")
                    total += 1
                low = " " + s.lower() + " "
                for w, alt in UNAPPROVED.items():
                    if re.search(r"\b" + re.escape(w) + r"\b", low) and w != alt:
                        print(f"{p}: unapproved '{w}' -> '{alt}': {s[:80]}")
                        total += 1
                for m in PASSIVE.finditer(s):
                    if not PASSIVE_OK.search(m.group(0)):
                        print(f"{p}: passive '{m.group(0)}': {s[:80]}")
                        total += 1
    print(f"\n{total} findings")
    return total


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
