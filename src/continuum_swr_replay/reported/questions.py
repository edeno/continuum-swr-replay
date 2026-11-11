"""The paper's questions and figures: which question each reported value answers,
and which figure shows it.

This module is the one place a question's wording, its place in the paper's
outline, its status and its figures are written. Each value in a summary
(:class:`~continuum_swr_replay.reported.values.Value`) names the questions it
answers in ``questions`` and the figures that show it in ``used_in``;
:func:`tag_problems` checks those names against the lists here, and
``scripts/reported/emit_reported_values.py`` writes ``paper/reported/questions.md``,
every question with its values.

A question is written as a question, not as its answer. Its id is its place in
the outline: its section's code and its number there (``"2d.2"``, section 2d's
second question).

The list organizes the writing by part. A question need not have a figure; a
figure tag names the figure the paper places a value in, which may be one whose
panels are not designed yet.

A summary file is named for what it holds, not for a part of the paper: a
question's part is where its values are reported.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """A section of the paper's outline.

    Attributes
    ----------
    code : str
        The prefix of its questions' ids (``"2d"``).
    part : int
        The part of the paper it belongs to.
    title : str
        What the section shows.
    """

    code: str
    part: int
    title: str


SECTIONS: tuple[Section, ...] = (
    Section("0", 0, "introduction"),
    Section("1", 1, "the model"),
    Section("2a", 2, "it works"),
    Section("2b", 2, "it finds events that other methods do not"),
    Section("2c", 2, "the events are real"),
    Section("2d", 2, "what the events are like"),
    Section("3", 3, "the events relate to behavior"),
    Section("4", 4, "the events relate to mPFC"),
)
"""The paper's outline, in order. Part 0 is the introduction's survey of published
replay studies (Supplementary Table S1)."""

SECTION_BY_CODE: dict[str, Section] = {s.code: s for s in SECTIONS}

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
    "decoder_validation": Figure(
        "Decoder validation", "scripts/figures/generate_supp_decoder_validation.py"
    ),
    "memoryless_comparison": Figure(
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
"""Every figure a value may name, by id: the main figures (``"2"``), with the panel
letters their scripts draw now, and the supplementary figures by their output
folder's name (``"memoryless_comparison"``). A value names one as
``"figure: <id>"`` or, for a lettered panel, ``"figure: <id><letter>"``
(``"figure: 2b"``). A value placed in a main figure whose panel is not designed
yet carries the unlettered tag (``"figure: 2"``), whether or not the script draws
it yet."""


@dataclass(frozen=True)
class Question:
    """One question the paper answers.

    Attributes
    ----------
    id : str
        Its section's code and its number in the section (``"2d.2"``).
    text : str
        The question.
    status : str
        One of :data:`STATUSES`.
    reason : str
        Why the status is not ``computed``; empty when it is.
    figures : tuple of str
        The figures or panels the paper places it in (``"2b"``,
        ``"memoryless_comparison"``); empty for a question no figure shows. Every
        figure tag on one of its values must be listed here.
    note : str
        An editorial note for the lead, if any.
    """

    id: str
    text: str
    status: str = "computed"
    reason: str = ""
    figures: tuple[str, ...] = ()
    note: str = ""

    @property
    def section(self) -> Section:
        """The section of the outline its id names."""
        return SECTION_BY_CODE[self.id.rsplit(".", 1)[0]]


QUESTIONS: tuple[Question, ...] = (
    Question(
        "0.1", "What detection and decoding parameters do published replay studies use?"
    ),
    Question(
        "1.1",
        "On simulated data, how often does the detector report an event where no replay was inserted, and how does it compare with the Widloski detector on replays of known shape?",
        figures=("1", "memoryless_comparison"),
    ),
    Question(
        "2a.1",
        "Is the decoder accurate, in sample and on held-out data?",
        status="not computed",
        reason="no summary value; Figure 2e and the decoder-validation supplement plot the in-sample decoding error",
        figures=("2e", "decoder_validation"),
    ),
    Question(
        "2a.2",
        "Do held-out likelihood and event-label stability favor the four-state model over the Continuous–Fragmented and reduced models?",
        status="not computed",
        reason="no script (held-out model selection)",
    ),
    Question(
        "2a.3",
        "On one real epoch, how many replays does the Widloski & Foster sliding-window detector call, and how many of those calls are decided by kept windows outside every piece?",
        figures=("2", "memoryless_comparison"),
    ),
    Question(
        "2a.4",
        "How often does the detector report non-local events on shuffled or thinned real spikes, and in windows where no replay is expected?",
        status="not computed",
        reason="no script (the false-positive rate on real data)",
    ),
    Question(
        "2a.5",
        "How do the non-SWR fraction and the close/remote split change across the non-local and SWR-z thresholds?",
        status="not computed",
        reason="no script (the threshold sweep)",
    ),
    Question(
        "2b.1",
        "Do non-local events occur without an SWR? (examples)",
        status="no numbers",
        reason="example events, one per overlap class",
        figures=("2a",),
    ),
    Question(
        "2b.2",
        "How many immobile non-local events overlap no SWR, and what fraction of them is that?",
        figures=("2", "2b"),
        note="The claims table reads these as the events SWR-triggered detection misses; that reading belongs in the Results.",
    ),
    Question(
        "2b.3",
        "How do SWR detection and non-local detection overlap: for each other detector, what share of its events overlaps no non-local event, how do those divide by decoder state, and, for the ones that do overlap, what share of their time bins does each state take?",
        figures=("2",),
    ),
    Question(
        "2b.4",
        "How do non-local event starts and ends differ from those of the matched SWR?",
        figures=("2d",),
    ),
    Question(
        "2b.5",
        "How is SWR z-score distributed at immobile Local and Non-Local time points, and how much of the non-local mass lies below the SWR-detection threshold?",
        figures=("3b",),
    ),
    Question(
        "2b.6",
        "Does the detector target the represented content rather than an LFP proxy?",
        status="no numbers",
        reason="framing (a candidate claim)",
        note="Wording written by the assistant from the claim; check it.",
    ),
    Question(
        "2c.1",
        "How do theta z-score and speed compare across SWR events, non-SWR events and running?",
        figures=("3b", "decoder_validation"),
    ),
    Question(
        "2c.2",
        "How do per-event spike count, MUA z-score and active tetrodes compare across non-SWR events, SWR events and matched immobile non-event windows?",
        figures=("2",),
    ),
    Question(
        "2c.3",
        "Do an event's spikes favor its decoded location over the animal's position, against matched windows and on held-out tetrodes?",
        figures=("2",),
    ),
    Question(
        "2c.4",
        "Where do event MAPs sit relative to the GPI peak, against matched quiet windows?",
    ),
    Question(
        "2c.5",
        "What spike evidence do matched immobile non-event windows have?",
        figures=("2",),
    ),
    Question(
        "2c.6",
        "Are close events produced by the extent of place fields?",
        status="not computed",
        reason="needs a re-decode with a wider Local kernel (a GPU subset)",
    ),
    Question(
        "2c.7",
        "Do remote events pass the spike-support tests (questions 2c.2, 2c.3)?",
        figures=("2",),
    ),
    Question(
        "2c.8",
        "How is the decoded distance from the animal distributed, by SWR status?",
        figures=("2",),
    ),
    Question(
        "2d.1",
        "Where does content sit relative to the animal and to the reward wells, SWR against non-SWR?",
        status="partly computed",
        reason="the own-well share waits for the wider-Local-kernel re-decode (question 2c.6)",
        figures=("2", "3"),
    ),
    Question(
        "2d.2",
        "How does replay speed differ between SWR and non-SWR events?",
        figures=("3c",),
    ),
    Question(
        "2d.3",
        "How does spatial extent differ between SWR and non-SWR events?",
        figures=("3",),
    ),
    Question(
        "2d.4",
        "How much does content move within an event (pauses, largest step, reversals), SWR against non-SWR?",
        figures=("3",),
    ),
    Question(
        "2d.5",
        "How concentrated is the posterior (peak, HPD coverage, multimodality), SWR against non-SWR?",
    ),
    Question("2d.6", "How does event duration differ between SWR and non-SWR events?"),
    Question(
        "3.1",
        "Which arms does content over-represent, and does that differ between SWR and non-SWR events once each kind's arm baseline is accounted for?",
        status="partly computed",
        reason="the re-decode with a wider Local kernel (question 2c.6) is pending",
        figures=("content_estimators", "swr_contrast"),
    ),
    Question(
        "3.2",
        "Is non-SWR content selective for recently rewarded arms (realized reward fraction)?",
        figures=("swr_contrast",),
    ),
    Question(
        "3.3",
        "How does the reward-history kernel of event occurrence (lag 1 and the slower lags) compare between SWR-coupled and non-SWR events?",
        figures=("swr_contrast",),
    ),
    Question(
        "3.4",
        "Does content correlate with the upcoming choice once behavior is in the model, with or without SWR?",
    ),
    Question("3.5", "Does non-SWR content predict staying or leaving?"),
    Question(
        "3.6",
        "How does SWR against non-SWR content depend on the arm's reward fraction (CS4) and, separately, on its value (CS5), as direct (length-adjusted) and total effects?",
        status="partly computed",
        reason="an interaction-block refit is pending",
        figures=("content_estimators", "swr_contrast"),
        note="The claim's \"high-value\" covers both; the claims table's dropped list calls the CS4 total effect avoidance of recently rewarded arms. Keep the two apart or merge them.",
    ),
    Question(
        "3.7",
        "Does content over-represent arms the animal is uncertain about, or arms of high value?",
        figures=("content_estimators", "uncertainty_content", "swr_contrast"),
    ),
    Question(
        "3.8",
        "Does the ripple power of events depend on the recent patch-switch rate, the previous trial's reward outcome or the post-outcome reward prediction error?",
    ),
    Question(
        "4.1",
        "Do non-SWR events modulate mPFC activity, against matched immobile non-event windows, and how does that compare with SWR events?",
        status="partly computed",
        reason="population inference and the SWR-controlled rung are pending, and equivalence is untested",
        figures=("4", "4a", "4b", "4c", "4d"),
    ),
    Question(
        "2b.7",
        "Are non-SWR events something other than sub-threshold SWRs?",
        status="dropped",
        reason="contradicted by the ripple-power bump at non-local onset; reframed as question 2b.5",
    ),
    Question(
        "3.9",
        "Are SWR and non-SWR events recruited differently by behavior?",
        status="dropped",
        reason="not shown as a gross recruitment difference (the peri-switch non-SWR profile is a null); superseded by the per-effect questions 3.1–3.7",
    ),
    Question(
        "3.10",
        "Does replay content predict the upcoming choice?",
        status="dropped",
        reason="invalidated by task autocorrelation; the one surviving thread, the non-SWR replay_frac term of question 3.4, is null over the behavioral baseline",
    ),
    Question(
        "2d.7",
        "Are non-SWR events a different kind of event, with a different function?",
        status="dropped",
        reason="not claimed: the events are characterized, and the SWR contrasts are descriptive, not categorical",
    ),
    Question(
        "2c.9",
        "Do tetrode holdout and interneuron exclusion, as main-text controls, show the events are real?",
        status="dropped",
        reason="folded into the likelihood test (question 2c.3)",
    ),
)
"""Every question, in the outline's order within each section; the dropped
questions last in theirs."""

QUESTION_IDS: dict[str, Question] = {q.id: q for q in QUESTIONS}


def parse_figure_tag(tag: str) -> tuple[str, str] | None:
    """The figure id and panel letter a ``used_in`` tag names, or None.

    ``"figure: 2b"`` is ``("2", "b")``; ``"figure: 2"`` and
    ``"figure: memoryless_comparison"`` have no panel (``""``). A tag that is not a
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


def tag_problems(questions: Iterable[str], used_in: Iterable[str]) -> list[str]:
    """What is wrong with a value's tags: an unknown question id, an unknown use
    or figure, a figure its question does not list, or a question whose status
    says it has no values (:data:`ANSWERED`). Empty when nothing is."""
    questions, used_in = tuple(questions), tuple(used_in)
    problems = [f"no question {q!r}" for q in questions if q not in QUESTION_IDS]
    figures = []
    for tag in used_in:
        if tag in USES:
            continue
        parsed = parse_figure_tag(tag)
        if parsed is None:
            problems.append(f"unknown use {tag!r}")
        else:
            figures.append("".join(parsed))
    for qid in questions:
        question = QUESTION_IDS.get(qid)
        if question is None:
            continue
        if question.status not in ANSWERED:
            problems.append(f"question {qid} is {question.status}, so has no values")
        for figure in figures:
            if figure not in question.figures:
                problems.append(f"question {qid} does not list figure {figure!r}")
    return problems
