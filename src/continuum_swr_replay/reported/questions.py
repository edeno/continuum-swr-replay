"""The paper's questions and figures: which question each reported value answers,
and which figure shows it.

This module is the one place a question's wording, its part of the paper, its
status and its figures are written. Each value in a summary
(:class:`~continuum_swr_replay.reported.values.Value`) names the questions it
answers in ``claims`` and the figures that show it in ``used_in``;
:func:`tag_problems` checks those names against the lists here, and
``scripts/reported/emit_reported_values.py`` writes ``paper/reported/questions.md``,
every question with its values.

A question is written as a question, not as its answer. The ids keep the claim
numbers the project has used, except where a claim was split or renamed (12 became
12a and 12b; the ripple-power model is 19h), so most existing tags stay valid.

The list organizes the writing by part. A question need not have a figure; a
figure tag names the figure the paper places a value in, which may be one whose
panels are not designed yet.

A summary file is named for what it holds, not for a part of the paper: a
question's part is where its values are reported.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SECTIONS: tuple[tuple[int, str], ...] = (
    (0, "introduction"),
    (1, "the model"),
    (2, "it works"),
    (2, "it finds events that other methods do not"),
    (2, "the events are real"),
    (2, "what the events are like"),
    (3, "the events relate to behavior"),
    (4, "the events relate to mPFC"),
)
"""The paper's outline, in order: the part and the section within it. Part 0 is the
introduction's survey of published replay studies (Supplementary Table S1)."""

STATUSES = ("computed", "partly computed", "not computed", "no numbers", "dropped")
"""A question's status. ``no numbers``: an example figure or a framing, which no
value answers. ``dropped``: considered and dropped, kept so the reason is not lost."""

ANSWERED = ("computed", "partly computed")
"""The statuses of a question that has values; a question of any other status has
none."""

USES = ("paper", "supplement")
"""The ``used_in`` tags other than a figure."""

FIGURE_PREFIX = "figure: "


@dataclass(frozen=True)
class Figure:
    """A figure a value can be shown in.

    Attributes
    ----------
    title : str
        As ``docs/figure-pipeline.md`` names it.
    script : str
        The script that draws it, from the repository root.
    panels : str
        Its panel letters, in order; empty for a figure whose panels are not
        lettered yet.
    """

    title: str
    script: str
    panels: str = ""


FIGURES: dict[str, Figure] = {
    "1": Figure(
        "The switching state-space model",
        "scripts/figures/generate_figure01.py",
        "abcd",
    ),
    "2": Figure(
        "Non-local events and SWRs", "scripts/figures/generate_figure02.py", "abcde"
    ),
    "3": Figure(
        "Non-local events outside SWRs", "scripts/figures/generate_figure03.py", "abcd"
    ),
    "4": Figure(
        "mPFC responses to SWR and non-SWR events",
        "scripts/figures/generate_figure04.py",
        "abcd",
    ),
    "decoder validation": Figure(
        "Decoder validation", "scripts/figures/generate_supp_decoder_validation.py"
    ),
    "memoryless comparison": Figure(
        "Our detector and the Widloski & Foster detector on simulated replays",
        "scripts/figures/generate_memoryless_comparison.py",
    ),
    "content_estimators": Figure(
        "Content models under three estimators",
        "scripts/content/compare_content_estimators.py",
    ),
    "uncertainty_content": Figure(
        "Content against reward-belief uncertainty",
        "scripts/figures/generate_uncertainty_content_figure.py",
    ),
    "swr_contrast": Figure(
        "SWR against non-SWR coefficients, rate and content",
        "scripts/figures/plot_swr_contrast_coefs.py",
    ),
}
"""Every figure a value may name: the main figures, with the panel letters their
scripts draw now, and the supplementary figures. A value names one as
``"figure: <id>"`` or, for a lettered panel, ``"figure: <id><letter>"``
(``"figure: 2b"``). A value placed in a main figure whose panel is not designed yet
carries the unlettered tag (``"figure: 2"``), whether or not the script draws it
yet."""


@dataclass(frozen=True)
class Question:
    """One question the paper answers.

    Attributes
    ----------
    id : str
        The claim number (``"15"``, ``"19a"``), or a new id for a question with no
        claim number.
    part : int
        The part of the paper, the first element of an entry of :data:`SECTIONS`.
    section : str
        The section within the part, as in :data:`SECTIONS`.
    text : str
        The question.
    status : str
        One of :data:`STATUSES`.
    reason : str
        Why the status is not ``computed``; empty when it is.
    figures : tuple of str
        The figures or panels the paper places it in (``"2b"``,
        ``"memoryless comparison"``); empty for a question no figure shows. Every
        figure tag on one of its values must be listed here.
    note : str
        An editorial note for the lead, if any.
    """

    id: str
    part: int
    section: str
    text: str
    status: str = "computed"
    reason: str = ""
    figures: tuple[str, ...] = ()
    note: str = ""


_WORKS = (2, "it works")
_FINDS = (2, "it finds events that other methods do not")
_REAL = (2, "the events are real")
_LIKE = (2, "what the events are like")
_BEHAVIOR = (3, "the events relate to behavior")
_MPFC = (4, "the events relate to mPFC")


def _q(where: tuple[int, str], id: str, text: str, **fields) -> Question:
    return Question(id=id, part=where[0], section=where[1], text=text, **fields)


QUESTIONS: tuple[Question, ...] = (
    _q(
        (0, "introduction"),
        "introduction",
        "What detection and decoding parameters do published replay studies use?",
    ),
    # Part 2: it works
    _q(
        _WORKS,
        "10",
        "Is the decoder accurate, in sample and on held-out data?",
        status="not computed",
        reason="no summary value; Figure 2e and the decoder-validation supplement "
        "plot the in-sample decoding error",
        figures=("2e", "decoder validation"),
    ),
    _q(
        _WORKS,
        "11",
        "Do held-out likelihood and event-label stability favor the four-state "
        "model over the Continuous–Fragmented and reduced models?",
        status="not computed",
        reason="no script (held-out model selection)",
    ),
    _q(
        _WORKS,
        "12a",
        "On simulated data, how often does the detector report an event where no "
        "replay was inserted, and how does it compare with the Widloski detector on "
        "replays of known shape?",
        figures=("memoryless comparison",),
    ),
    _q(
        _WORKS,
        "12b",
        "How often does the detector report non-local events on shuffled or thinned "
        "real spikes, and in windows where no replay is expected?",
        status="not computed",
        reason="no script (the false-positive rate on real data)",
    ),
    _q(
        _WORKS,
        "3",
        "How do the non-SWR fraction and the close/remote split change across the "
        "non-local and SWR-z thresholds?",
        status="not computed",
        reason="no script (the threshold sweep)",
    ),
    # Part 2: it finds events that other methods do not
    _q(
        _FINDS,
        "1",
        "Do non-local events occur without an SWR? (examples)",
        status="no numbers",
        reason="example events, one per overlap class",
        figures=("2a",),
    ),
    _q(
        _FINDS,
        "2",
        "What fraction of immobile non-local events overlaps no SWR?",
        figures=("2", "2b"),
    ),
    _q(
        _FINDS,
        "21",
        "How many detected non-local events overlap no SWR?",
        figures=("2", "2b"),
        note="The claims table reads these as the events SWR-triggered detection "
        "misses; that reading belongs in the Results.",
    ),
    _q(
        _FINDS,
        "4",
        "How do SWR detection and non-local detection overlap, in events and in "
        "time points?",
        figures=("2",),
    ),
    _q(
        _FINDS,
        "23",
        "For each other detector, what share of its events overlaps no non-local "
        "event, and how do those divide by decoder state; for the ones that do "
        "overlap, what share of their time bins does each state take?",
        figures=("2",),
    ),
    _q(
        _FINDS,
        "22",
        "How do non-local event starts and ends differ from those of the matched SWR?",
        figures=("2d",),
    ),
    _q(
        _FINDS,
        "24",
        "How is SWR z-score distributed at immobile Local and Non-Local time points, "
        "and how much of the non-local mass lies below the SWR-detection threshold?",
        figures=("3b",),
    ),
    _q(
        _FINDS,
        "25",
        "Does the detector target the represented content rather than an LFP proxy?",
        status="no numbers",
        reason="framing (a candidate claim)",
        note="Wording written by the assistant from the claim; check it.",
    ),
    # Part 2: the events are real
    _q(
        _REAL,
        "5",
        "How do theta z-score and speed compare across SWR events, non-SWR events "
        "and running?",
        figures=("3b", "decoder validation"),
    ),
    _q(
        _REAL,
        "6",
        "How do per-event spike count, MUA z-score and active tetrodes compare "
        "across non-SWR events, SWR events and matched immobile non-event windows?",
        figures=("2",),
    ),
    _q(
        _REAL,
        "7",
        "Do an event's spikes favor its decoded location over the animal's "
        "position, against matched windows and on held-out tetrodes?",
        figures=("2",),
    ),
    _q(
        _REAL,
        "7b",
        "Where do event MAPs sit relative to the GPI peak, against matched quiet "
        "windows?",
    ),
    _q(
        _REAL,
        "7c",
        "What spike evidence do matched immobile non-event windows have?",
        figures=("2",),
    ),
    _q(
        _REAL,
        "8",
        "Are close events produced by the extent of place fields?",
        status="not computed",
        reason="needs a re-decode with a wider Local kernel (a GPU subset)",
    ),
    _q(
        _REAL,
        "9",
        "Do remote events pass the spike-support tests (6, 7)?",
        figures=("2",),
    ),
    _q(
        _REAL,
        "13",
        "How is the decoded distance from the animal distributed, by SWR status?",
        figures=("2",),
    ),
    # Part 2: what the events are like
    _q(
        _LIKE,
        "14",
        "Where does content sit relative to the animal and to the reward wells, SWR "
        "against non-SWR?",
        status="partly computed",
        reason="the own-well share waits for the wider-Local-kernel re-decode "
        "(question 8)",
        figures=("2", "3"),
    ),
    _q(
        _LIKE,
        "15",
        "How does replay speed differ between SWR and non-SWR events?",
        figures=("3c",),
    ),
    _q(
        _LIKE,
        "16",
        "How does spatial extent differ between SWR and non-SWR events?",
        figures=("3",),
    ),
    _q(
        _LIKE,
        "16b",
        "How much does content move within an event (pauses, largest step, "
        "reversals), SWR against non-SWR?",
        figures=("3",),
    ),
    _q(
        _LIKE,
        "17",
        "How concentrated is the posterior (peak, HPD coverage, multimodality), SWR "
        "against non-SWR?",
    ),
    _q(
        _LIKE,
        "18",
        "How does event duration differ between SWR and non-SWR events?",
    ),
    # Part 3: behavior
    _q(
        _BEHAVIOR,
        "19a",
        "Which arms does content over-represent, and does that differ between SWR "
        "and non-SWR events once each kind's arm baseline is accounted for?",
        status="partly computed",
        reason="the re-decode with a wider Local kernel (question 8) is pending",
        figures=("content_estimators", "swr_contrast"),
    ),
    _q(
        _BEHAVIOR,
        "19b",
        "Is non-SWR content selective for recently rewarded arms (realized reward "
        "fraction)?",
        figures=("swr_contrast",),
    ),
    _q(
        _BEHAVIOR,
        "19c",
        "How does the reward-history kernel of event occurrence (lag 1 and the "
        "slower lags) compare between SWR-coupled and non-SWR events?",
        figures=("swr_contrast",),
    ),
    _q(
        _BEHAVIOR,
        "19d",
        "Does content correlate with the upcoming choice once behavior is in the "
        "model, with or without SWR?",
    ),
    _q(_BEHAVIOR, "19e", "Does non-SWR content predict staying or leaving?"),
    _q(
        _BEHAVIOR,
        "19f",
        "How does SWR against non-SWR content depend on the arm's reward fraction "
        "(CS4) and, separately, on its value (CS5), as direct (length-adjusted) and "
        "total effects?",
        status="partly computed",
        reason="an interaction-block refit is pending",
        figures=("content_estimators", "swr_contrast"),
        note="The claim's \"high-value\" covers both; the claims table's dropped list "
        "calls the CS4 total effect avoidance of recently rewarded arms. Keep the two "
        "apart or merge them.",
    ),
    _q(
        _BEHAVIOR,
        "19g",
        "Does content over-represent arms the animal is uncertain about, or arms of "
        "high value?",
        figures=("content_estimators", "uncertainty_content", "swr_contrast"),
    ),
    _q(
        _BEHAVIOR,
        "19h",
        "Does the ripple power of events depend on the recent patch-switch rate, the "
        "previous trial's reward outcome or the post-outcome reward prediction error?",
    ),
    # Part 4: mPFC
    _q(
        _MPFC,
        "20",
        "Do non-SWR events modulate mPFC activity, against matched immobile "
        "non-event windows, and how does that compare with SWR events?",
        status="partly computed",
        reason="population inference and the SWR-controlled rung are pending, and "
        "equivalence is untested",
        figures=("4", "4a", "4b", "4c", "4d"),
    ),
    # Considered and dropped
    _q(
        _FINDS,
        "dropped-subthreshold",
        "Are non-SWR events something other than sub-threshold SWRs?",
        status="dropped",
        reason="contradicted by the ripple-power bump at non-local onset; reframed "
        "as question 24",
    ),
    _q(
        _BEHAVIOR,
        "dropped-recruitment",
        "Are SWR and non-SWR events recruited differently by behavior?",
        status="dropped",
        reason="not shown as a gross recruitment difference (the peri-switch non-SWR "
        "profile is a null); superseded by the per-effect questions 19a–19g",
    ),
    _q(
        _BEHAVIOR,
        "dropped-choice",
        "Does replay content predict the upcoming choice?",
        status="dropped",
        reason="invalidated by task autocorrelation; the one surviving thread, the "
        "non-SWR replay_frac term of question 19d, is null over the behavioral "
        "baseline",
    ),
    _q(
        _LIKE,
        "dropped-kind",
        "Are non-SWR events a different kind of event, with a different function?",
        status="dropped",
        reason="not claimed: the events are characterized, and the SWR contrasts are "
        "descriptive, not categorical",
    ),
    _q(
        _REAL,
        "dropped-reality-controls",
        "Do tetrode holdout and interneuron exclusion, as main-text controls, "
        "show the events are real?",
        status="dropped",
        reason="folded into the likelihood test (question 7)",
    ),
)
"""Every question, in the outline's order within each section."""

QUESTION_IDS: dict[str, Question] = {q.id: q for q in QUESTIONS}


def parse_figure_tag(tag: str) -> tuple[str, str] | None:
    """The figure id and panel letter a ``used_in`` tag names, or None.

    ``"figure: 2b"`` is ``("2", "b")``; ``"figure: 2"`` and
    ``"figure: memoryless comparison"`` have no panel (``""``). A tag that is not a
    figure tag, or names no figure in :data:`FIGURES`, is None.
    """
    if not tag.startswith(FIGURE_PREFIX):
        return None
    name = tag[len(FIGURE_PREFIX) :]
    if name in FIGURES:
        return name, ""
    figure, panel = name[:-1], name[-1:]
    if figure in FIGURES and panel and panel in FIGURES[figure].panels:
        return figure, panel
    return None


def tag_problems(claims: Iterable[str], used_in: Iterable[str]) -> list[str]:
    """What is wrong with a value's tags: an unknown question id, an unknown use
    or figure, a figure its question does not list, or a question whose status
    says it has no values (:data:`ANSWERED`). Empty when nothing is."""
    claims, used_in = tuple(claims), tuple(used_in)
    problems = [f"no question {c!r}" for c in claims if c not in QUESTION_IDS]
    figures = []
    for tag in used_in:
        if tag in USES:
            continue
        parsed = parse_figure_tag(tag)
        if parsed is None:
            problems.append(f"unknown use {tag!r}")
        else:
            figures.append("".join(parsed))
    for claim in claims:
        question = QUESTION_IDS.get(claim)
        if question is None:
            continue
        if question.status not in ANSWERED:
            problems.append(f"question {claim} is {question.status}, so has no values")
        for figure in figures:
            if figure not in question.figures:
                problems.append(f"question {claim} does not list figure {figure!r}")
    return problems
