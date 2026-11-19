"""Find, or fix, British spellings in the repository's tracked text.

The repository writes American English, in prose and in names alike
(``analyzed_epochs``, ``color``, ``center``, ``behavior``). This script reads
every tracked text file and reports each British spelling it finds, with the
American one; ``--fix`` rewrites them in place. ``tests/repository/
test_american_spelling.py`` runs the report, so a British spelling fails the
suite.

Words are matched inside names too: ``n_analysed``, ``plotColour`` and
``NOT_ANALYSED`` are split at underscores and at camelCase humps, and each
part is checked. The case of each part is kept.

What is read, and how much of it:

* vendored and quoted text is not read: the agent skills (``.claude/skills/``),
  the reference code of other labs (``reference/``), the bibliography (titles
  are spelled as published) and ``LICENSE``;
* historical records -- the paths ``tests/repository/test_doc_paths_resolve.py``
  calls historical -- are read as prose only: their code spans and fenced
  blocks name the code as it was then, and are left as written;
* a notebook's cell sources are read, not its outputs;
* everything else is read whole.

``--fix`` does not touch the summaries (``paper/reported/``) or the macros
(``paper/reported_values.tex``): they are rebuilt (``make summaries``,
``make reported-values``), not edited.

Some British spellings are names this repository does not own, and are kept
(:data:`KEPT`): matplotlib's ``colColours`` keyword and ``Greys`` colormap, and
the ``n_analysed`` key of the PFC cohort records already on disk.

Usage::

    uv run --frozen python scripts/checks/american_spelling.py        # report
    uv run --frozen python scripts/checks/american_spelling.py --fix  # rewrite
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- the words -------------------------------------------------------------

#: Roots whose ``-ise`` forms are British (``normal`` -> ``normalise``).
_ISE_ROOTS = (
    "amort anonym apolog atom author binar canonical capital categor central "
    "character conceptual contextual critic crystall custom denormal depolar "
    "deserial desensit destabil desynchron digit discret emphas equal factor "
    "familiar fertil final general global harmon hospital hyperpolar hypothes "
    "ideal initial item lateral legal legitim lemmat linear local marginal "
    "material maxim memor metabol minim mobil modern neutral normal "
    "operational optim organ orthogonal oxid parameter parametr penal personal "
    "polar popular priorit quant random raster rational real recogn reorgan "
    "renormal reparameter reparametr regular sanit scrutin sensit serial social "
    "special stabil standard steril stigmat summar symbol synchron synthes "
    "theor token trivial util vector visual vocal"
).split()
_ISE_SUFFIXES = {
    "ise": "ize",
    "ised": "ized",
    "ises": "izes",
    "ising": "izing",
    "isation": "ization",
    "isations": "izations",
    "iser": "izer",
    "isers": "izers",
    "isable": "izable",
}
#: Roots whose ``-yse`` forms are British. ``analyses`` is left out: it is
#: also the plural of ``analysis`` (see :data:`_VERB_ANALYSES`).
_YSE_ROOTS = ("anal", "catal", "dial", "electrol", "hydrol", "paral")
_YSE_SUFFIXES = {
    "yse": "yze",
    "ysed": "yzed",
    "ysing": "yzing",
    "yser": "yzer",
    "ysers": "yzers",
    "ysable": "yzable",
}
_OUR_BASES = (
    "colour behaviour neighbour favour honour flavour labour rigour vigour "
    "humour odour harbour rumour vapour armour endeavour tumour savour "
    "splendour candour fervour ardour valour clamour parlour"
).split()
_OUR_SUFFIXES = (
    "",
    "s",
    "ed",
    "ing",
    "ings",
    "al",
    "ally",
    "able",
    "ably",
    "ite",
    "ites",
    "ful",
    "less",
    "hood",
    "hoods",
    "er",
    "ers",
    "ist",
    "ists",
    "map",
    "maps",
    "bar",
    "bars",
    "blind",
    "way",
    "ways",
)
_OTHER = {
    # -re
    "centre": "center",
    "centres": "centers",
    "centred": "centered",
    "centring": "centering",
    "centreline": "centerline",
    "centrelines": "centerlines",
    "epicentre": "epicenter",
    "metre": "meter",
    "metres": "meters",
    "centimetre": "centimeter",
    "centimetres": "centimeters",
    "millimetre": "millimeter",
    "millimetres": "millimeters",
    "micrometre": "micrometer",
    "micrometres": "micrometers",
    "nanometre": "nanometer",
    "nanometres": "nanometers",
    "kilometre": "kilometer",
    "kilometres": "kilometers",
    "fibre": "fiber",
    "fibres": "fibers",
    "litre": "liter",
    "litres": "liters",
    "millilitre": "milliliter",
    "millilitres": "milliliters",
    "theatre": "theater",
    "theatres": "theaters",
    "calibre": "caliber",
    "spectre": "specter",
    "lustre": "luster",
    "meagre": "meager",
    "sombre": "somber",
    # -ll-
    "labelled": "labeled",
    "labelling": "labeling",
    "labeller": "labeler",
    "labellers": "labelers",
    "mislabelled": "mislabeled",
    "mislabelling": "mislabeling",
    "relabelled": "relabeled",
    "relabelling": "relabeling",
    "unlabelled": "unlabeled",
    "modelled": "modeled",
    "modelling": "modeling",
    "modeller": "modeler",
    "modellers": "modelers",
    "unmodelled": "unmodeled",
    "travelled": "traveled",
    "travelling": "traveling",
    "traveller": "traveler",
    "travellers": "travelers",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "totalled": "totaled",
    "totalling": "totaling",
    "signalled": "signaled",
    "signalling": "signaling",
    "channelled": "channeled",
    "channelling": "channeling",
    "levelled": "leveled",
    "levelling": "leveling",
    "tunnelled": "tunneled",
    "tunnelling": "tunneling",
    "fuelled": "fueled",
    "fuelling": "fueling",
    "marshalled": "marshaled",
    "marshalling": "marshaling",
    "dialled": "dialed",
    "dialling": "dialing",
    "equalled": "equaled",
    "equalling": "equaling",
    "rivalled": "rivaled",
    "panelled": "paneled",
    "jewellery": "jewelry",
    # -ogue, -ence
    "analogue": "analog",
    "analogues": "analogs",
    "catalogue": "catalog",
    "catalogues": "catalogs",
    "catalogued": "cataloged",
    "cataloguing": "cataloging",
    "defence": "defense",
    "defences": "defenses",
    "offence": "offense",
    "offences": "offenses",
    "pretence": "pretense",
    "licence": "license",
    "licences": "licenses",
    # ae, oe
    "anaesthesia": "anesthesia",
    "anaesthetic": "anesthetic",
    "anaesthetics": "anesthetics",
    "anaesthetised": "anesthetized",
    "anaesthetized": "anesthetized",
    "haemodynamic": "hemodynamic",
    "haemoglobin": "hemoglobin",
    "haematoxylin": "hematoxylin",
    "oedema": "edema",
    "oestrus": "estrus",
    "oestrous": "estrous",
    "oestrogen": "estrogen",
    "foetal": "fetal",
    "foetus": "fetus",
    "paediatric": "pediatric",
    "aetiology": "etiology",
    "oesophagus": "esophagus",
    "orthopaedic": "orthopedic",
    "manoeuvre": "maneuver",
    "manoeuvres": "maneuvers",
    "manoeuvring": "maneuvering",
    # gray, and matplotlib's color names that have both spellings
    "grey": "gray",
    "greys": "grays",
    "greyed": "grayed",
    "greying": "graying",
    "greyish": "grayish",
    "greyscale": "grayscale",
    "lightgrey": "lightgray",
    "darkgrey": "darkgray",
    "dimgrey": "dimgray",
    "slategrey": "slategray",
    "lightslategrey": "lightslategray",
    "darkslategrey": "darkslategray",
    # the rest
    "judgement": "judgment",
    "judgements": "judgments",
    "acknowledgement": "acknowledgment",
    "acknowledgements": "acknowledgments",
    "artefact": "artifact",
    "artefacts": "artifacts",
    "artefactual": "artifactual",
    "ageing": "aging",
    "sceptic": "skeptic",
    "sceptics": "skeptics",
    "sceptical": "skeptical",
    "scepticism": "skepticism",
    "programme": "program",
    "programmes": "programs",
    "whilst": "while",
    "amongst": "among",
    "learnt": "learned",
    "spelt": "spelled",
    "dreamt": "dreamed",
    "towards": "toward",
    "afterwards": "afterward",
    "backwards": "backward",
    "outwards": "outward",
    "inwards": "inward",
    "upwards": "upward",
    "downwards": "downward",
    "focussed": "focused",
    "focusses": "focuses",
    "focussing": "focusing",
    "benefitted": "benefited",
    "benefitting": "benefiting",
    "fulfil": "fulfill",
    "fulfils": "fulfills",
    "fulfilment": "fulfillment",
    "enrol": "enroll",
    "enrols": "enrolls",
    "enrolment": "enrollment",
    "instalment": "installment",
    "skilful": "skillful",
    "wilful": "willful",
    "speciality": "specialty",
    "maths": "math",
    "orientated": "oriented",
    "cosy": "cozy",
    "mould": "mold",
    "plough": "plow",
    "aluminium": "aluminum",
    "practise": "practice",
    "practised": "practiced",
    "practising": "practicing",
    "enquire": "inquire",
    "enquiry": "inquiry",
    "enquiries": "inquiries",
}


def _word_map() -> dict[str, str]:
    words = dict(_OTHER)
    for root in _ISE_ROOTS:
        for british, american in _ISE_SUFFIXES.items():
            words[root + british] = root + american
        words["un" + root + "ised"] = "un" + root + "ized"
    for root in _YSE_ROOTS:
        for british, american in _YSE_SUFFIXES.items():
            words[root + british] = root + american
        words["un" + root + "ysed"] = "un" + root + "yzed"
    for base in _OUR_BASES:
        for suffix in _OUR_SUFFIXES:
            words[base + suffix] = base[:-2] + "r" + suffix
    for base in ("favourable", "favourably"):
        words["un" + base] = "un" + words[base]
    return words


#: British spelling -> American, both lower case.
BRITISH_TO_AMERICAN: dict[str, str] = _word_map()

#: British spellings kept, because this repository does not own the name:
#: matplotlib's ``table`` keywords and its ColorBrewer colormap (the ``Grays``
#: alias needs matplotlib 3.9, above this package's floor), XLA's ``CANCELLED``
#: status in the JAX errors quoted, and the key the PFC cohort records already
#: on disk carry (``run_pfc_analysis_bandit.py`` writes it and
#: ``reported/mpfc/population.py`` reads it; renaming it means migrating those
#: records).
KEPT = (
    "colColours",
    "rowColours",
    "cellColours",
    "Greys",
    "CANCELLED",
    '"n_analysed"',
)

# ``analyses`` is a British verb (the register analyses 242 epochs) and an
# American plural noun (these analyses); only the verb uses this repository
# writes are matched.
_VERB_ANALYSES = re.compile(
    r"(?:(?<=register )|(?<=register_)|(?<=register that )|(?<=silently )"
    r"|(?<=\.csv ))(analyses|Analyses)(?![a-z])"
)


#: A run of letters: names split here at underscores and digits.
_LETTERS = re.compile(r"[A-Za-z]+")
#: The parts of a run of letters: ``HTMLColour`` is ``HTML`` and ``Colour``,
#: ``MpfcAnalysedEpochs`` is ``Mpfc``, ``Analysed`` and ``Epochs``.
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
_KEPT = re.compile(
    "|".join(rf"(?<![A-Za-z]){re.escape(name)}(?![a-z])" for name in KEPT)
)


def _american_part(part: str) -> str:
    american = BRITISH_TO_AMERICAN.get(part.lower())
    if american is None:
        return part
    if part.isupper() and len(part) > 1:
        return american.upper()
    if part[0].isupper():
        return american.capitalize()
    return american


def _american(match: re.Match) -> str:
    letters = match.group(0)
    if letters.islower():
        return BRITISH_TO_AMERICAN.get(letters, letters)
    return "".join(_american_part(part) for part in _PARTS.findall(letters))


def _masked(text: str, pattern: re.Pattern, transform) -> str:
    """``transform`` applied to ``text`` outside ``pattern``'s matches."""
    pieces, last = [], 0
    for match in pattern.finditer(text):
        pieces.append(transform(text[last : match.start()]))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(transform(text[last:]))
    return "".join(pieces)


def _americanize_words(text: str) -> str:
    text = _VERB_ANALYSES.sub(
        lambda m: "analyzes" if m.group(1) == "analyses" else "Analyzes", text
    )
    return _LETTERS.sub(_american, text)


def americanize(text: str) -> str:
    """``text`` with its British spellings made American, kept names aside."""
    return _masked(text, _KEPT, _americanize_words)


#: Markdown code: fenced blocks, then inline spans of one or two backticks.
_MARKDOWN_CODE = re.compile(
    r"^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$|``[^`\n]+``|`[^`\n]+`",
    re.MULTILINE | re.DOTALL,
)


def americanize_prose(markdown: str) -> str:
    """``americanize`` outside the Markdown code spans and fenced blocks."""
    return _masked(markdown, _MARKDOWN_CODE, americanize)


def americanize_notebook(raw: str) -> str:
    """``americanize`` on a notebook's cell sources; outputs are left alone."""
    notebook = json.loads(raw)
    changed = False
    for cell in notebook["cells"]:
        source = cell["source"]
        text = source if isinstance(source, str) else "".join(source)
        fixed = americanize(text)
        if fixed != text:
            changed = True
            cell["source"] = (
                fixed if isinstance(source, str) else fixed.splitlines(keepends=True)
            )
    if not changed:
        return raw
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


# --- the files ---------------------------------------------------------------

#: Not read: vendored or quoted text, and this script and its test, which
#: spell the British words on purpose.
_NOT_READ_PREFIXES = (".claude/skills/", "reference/")
_NOT_READ = {
    "paper/references.bib",
    "LICENSE",
    "uv.lock",
    "scripts/checks/american_spelling.py",
    "tests/repository/test_american_spelling.py",
}
_BINARY_SUFFIXES = {".png", ".pdf", ".jpg", ".jpeg", ".svg", ".gif", ".ico"}

#: Historical records, read as prose: the paths that
#: ``tests/repository/test_doc_paths_resolve.py`` does not check.
_RECORD_PREFIXES = (
    ".claude/docs/plans/",
    ".claude/docs/reviews/",
    "docs/known_issues/",
    "docs/history/",
)
_DATED_DESIGN = re.compile(r"docs/design/\d{4}-\d{2}-\d{2}-[^/]*\.md")

#: Rebuilt, not edited: ``--fix`` leaves them for ``make summaries``.
_REBUILT_PREFIXES = ("paper/reported/", "paper/reported_values.tex")


def read_as(path: str) -> str | None:
    """How a tracked path is read: ``"whole"``, ``"prose"``, ``"notebook"``,
    or ``None`` when it is not read."""
    if (
        path.startswith(_NOT_READ_PREFIXES)
        or path in _NOT_READ
        or Path(path).suffix.lower() in _BINARY_SUFFIXES
        or Path(path).name == ".DS_Store"
    ):
        return None
    if path.endswith(".ipynb"):
        return "notebook"
    if path.endswith(".md") and (
        path.startswith(_RECORD_PREFIXES) or _DATED_DESIGN.fullmatch(path)
    ):
        return "prose"
    return "whole"


def americanized(path: str, text: str) -> str:
    """The American version of one tracked file's text."""
    mode = read_as(path)
    if mode == "notebook":
        return americanize_notebook(text)
    if mode == "prose":
        return americanize_prose(text)
    return americanize(text)


def tracked_files(repo: Path = REPO) -> list[str]:
    """The tracked paths, relative to ``repo``, that this script reads."""
    paths = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=True
    ).stdout.decode()
    return [p for p in paths.split("\0") if p and read_as(p) is not None]


def british_spellings(repo: Path = REPO) -> list[str]:
    """``path:line: british -> american`` for each British spelling left."""
    found = []
    for path in tracked_files(repo):
        file = repo / path
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        fixed = americanized(path, text)
        if fixed == text:
            continue
        if read_as(path) == "notebook":
            found.append(f"{path}: British spellings in its cell sources")
            continue
        for number, (line, new) in enumerate(
            zip(text.splitlines(), fixed.splitlines(), strict=True), start=1
        ):
            if line != new:
                found.append(f"{path}:{number}: {line.strip()} -> {new.strip()}")
    return found


def fix(repo: Path = REPO) -> list[str]:
    """Rewrite the British spellings in place; return the paths changed."""
    changed = []
    for path in tracked_files(repo):
        file = repo / path
        if path.startswith(_REBUILT_PREFIXES) or not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        fixed = americanized(path, text)
        if fixed != text:
            file.write_text(fixed, encoding="utf-8")
            changed.append(path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fix", action="store_true", help="rewrite the British spellings in place"
    )
    args = parser.parse_args(argv)
    if args.fix:
        changed = fix()
        print(f"rewrote {len(changed)} files")
        return 0
    found = british_spellings()
    print("\n".join(found) if found else "no British spellings")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
