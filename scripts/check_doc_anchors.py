"""Validate that every internal ``(#anchor)`` in a markdown doc resolves to a heading.

Renaming a section silently breaks its inbound links, which is invisible until someone
clicks one. Cheap to check, so check it.

    python scripts/check_doc_anchors.py docs/exps/lm_head/results_lm_head.md
"""

import re
import sys


def slug(heading):
    """GitHub's anchor slug: lowercase, drop punctuation, then EACH space -> one hyphen.

    The per-space rule matters: "Part 1 — Reducing ..." drops the em-dash but keeps both
    surrounding spaces, so the anchor is ``part-1--reducing-...`` with a double hyphen.
    Collapsing whitespace instead reports those correct links as broken.
    """
    s = re.sub(r"[^\w\s-]", "", heading.lower(), flags=re.UNICODE)
    return s.strip().replace(" ", "-")


def main(paths):
    bad_total = 0
    for path in paths:
        t = open(path).read()
        heads = {slug(h) for h in re.findall(r"^#+ (.+)$", t, re.M)}
        links = sorted(set(re.findall(r"\(#([^)]+)\)", t)))
        bad = [a for a in links if a not in heads]
        print(f"{path}: {len(links)} internal links, {len(heads)} headings, "
              f"{len(bad)} unresolved")
        for a in bad:
            print(f"    UNRESOLVED  #{a}")
        bad_total += len(bad)
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["docs/exps/lm_head/results_lm_head.md"]))
