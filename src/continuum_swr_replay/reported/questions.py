"""The paper's questions and figures: which question each reported value answers,
and which figure shows it.

This module is the one place a question's wording, its place in the paper's
outline, its status and its figures are written. Each value in a summary
(:class:`~continuum_swr_replay.reported.values.Value`) names the questions it
answers in ``questions`` and the figures that show it in ``figures``;
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
        "abcde",
    ),
    "2": Figure(
        "Non-local events and SWRs",
        "scripts/figures/generate_figure02.py",
        "abcdefghij",
    ),
    "3": Figure(
        "What SWR and no-SWR events are like",
        "scripts/figures/generate_figure03.py",
        "abcdef",
    ),
    "5": Figure(
        "mPFC responses to SWR and non-SWR events",
        "scripts/figures/generate_figure05.py",
        "abcd",
    ),
    "decoder_validation": Figure(
        "Decoder validation", "scripts/figures/generate_supp_decoder_validation.py"
    ),
    "S1": Figure(
        "Our detector and the Widloski & Foster detector on simulated replays",
        "scripts/figures/generate_supp_figure01.py",
        "abcd",
    ),
    "S2": Figure(
        "The decoder's states and MUA in No-Spike; the event rule's sensitivity and "
        "the timing of events against their SWRs; non-local time against MUA, the "
        "posterior's spread, the held-out tetrode gain and shared content; the "
        "events' peak ripple power and the gain outside SWRs away from the SWRs",
        "scripts/figures/generate_supp_figure02.py",
        "abcdefghij",
    ),
    "S3": Figure(
        "Slow and fast gamma during SWR and no-SWR events",
        "scripts/figures/generate_supp_figure03.py",
        "ab",
    ),
    "content_estimators": Figure(
        "Content models under three estimators",
        "scripts/replay_models/content/compare_content_estimators.py",
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
"""Every figure a value may name, by id: the main figures (``"2"``) and the
supplement of each (``"S2"``, ``scripts/figures/generate_supp_figure02.py``),
with the panel letters their scripts draw now, and the other supplementary
figures by their output folder's name (``"decoder_validation"``). A value names
one by its id or, for a lettered panel, its id and the letter (``"2b"``,
``"S2b"``). A value placed in a main
figure whose panel is not designed yet names the figure without a letter
(``"2"``), whether or not the script draws it yet."""


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
        ``"S1"``); empty for a question no figure shows. Every
        figure one of its values names must be listed here.
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
        "On simulated replays of known shape, how often does each detector report the replay and how well does it place its start and end, the state-space detector against the Widloski & Foster sliding-window detector?",
        figures=("1", "S1"),
    ),
    Question(
        "1.2",
        "On simulated activity that is not one clean replay (rest with nothing added, events that represent no place, events whose place jumps, replays in two parts), what does each detector report?",
        note="The controls sweep. No figure shows it; it holds the simulation's false-positive evidence.",
    ),
    Question(
        "1.3",
        "On simulated data, where the truth of every event is known, does the held-out spike test support events with spatial content and not events without it?",
        note="The held-out sweep. It calibrates the held-out test that question 2c.3 applies to the recordings. No figure shows it.",
    ),
    Question(
        "2a.1",
        "Is the decoder accurate, in sample and on held-out data?",
        status="partly computed",
        reason="in sample only: no script measures the held-out decoding error",
        figures=("decoder_validation",),
    ),
    Question(
        "2a.3",
        "On one real epoch, how many replays does the Widloski & Foster sliding-window detector call, and how many of those calls are decided by kept windows outside every piece?",
        figures=("S1",),
    ),
    Question(
        "2a.5",
        "How do the non-SWR fraction and the close/remote split change across the non-local and SWR-z thresholds?",
        status="partly computed",
        reason="the non-SWR fraction is swept over the non-local threshold, the minimum duration and the SWR threshold; the close/remote split is not",
        figures=("S2c", "S2i"),
    ),
    Question(
        "2a.6",
        "How does the decoder divide time among its four states while the animal moves and while it is still?",
        figures=("2a", "decoder_validation"),
        note="Added with the decoder-validation figure's state-fraction values; check its wording and place.",
    ),
    Question(
        "2a.7",
        "Do held-out spikes support what each decoder state claims: the animal's position in Local, a remote position in the Non-Local states, silence in No-Spike?",
        figures=("S2a",),
        note="The held-out tetrode test on confident state bouts. It validates the model's states; whether the events are supported is question 2c.3.",
    ),
    Question(
        "2b.1",
        "Do non-local events occur without an SWR? (examples)",
        status="no numbers",
        reason="example events, one per overlap class",
        figures=("2b",),
    ),
    Question(
        "2b.2",
        "How many immobile non-local events overlap no SWR, and what fraction of them is that?",
        figures=("2", "2d"),
        note="The retired claims table (in git history) read these as the events SWR-triggered detection misses; that reading belongs in the Results.",
    ),
    Question(
        "2b.3",
        "How do SWR detection and non-local detection overlap: for each other detector, what share of its events overlaps no non-local event, how do those divide by decoder state, and, for the ones that do overlap, what share of their time bins does each state take?",
        figures=("2", "2c", "2e"),
    ),
    Question(
        "2b.4",
        "How do non-local event starts and ends differ from those of the matched SWR?",
        figures=("2f", "S2d"),
        note="Figure 2f draws each bound's offset; Figure 2's supplement, panel S2d, is the timing_* values: the share of events in each combination of start and end against the SWR's (earlier, together within 25 ms, later).",
    ),
    Question(
        "2b.5",
        "How is SWR z-score distributed at immobile Local and Non-Local time points, and how much of the non-local mass lies below the SWR-detection threshold?",
        figures=("2g", "2i", "S2e", "S2i"),
    ),
    Question(
        "2c.3",
        "Do an event's spikes favor its decoded location over the animal's position (against matched windows and on held-out tetrodes), and do held-out spikes favor a decoder that may be non-local outside SWRs over one that is non-local only in SWRs or never?",
        figures=("2", "2j", "S2g", "S2h", "S2j"),
        note="Two lines of evidence. The spike evidence: the matched-window, likelihood-advantage and held-out tetrode values (the held-out tetrode score of Figure 2's supplement, panel S2g, is the *_heldout_gain_*, *_heldout_spiking_gain_* and *_heldout_control_gain_* values; panel S2h, shared content, is the *_shared_content_* values: the held-out agreement per spike of the events by the spread of the other tetrodes' estimate, in bins fixed in advance). The decoder comparison: the restriction_* values compare the local-only, SWR-only and full decoders by held-out predictive score, over immobile time (restriction_immobile_* and restriction_outside_swr_*; Figure 2j draws each decoder's gain over the local-only decoder and the full minus the SWR-only decoder over all immobile time and outside SWRs; Figure 2's supplement draws that gain outside SWRs away from the SWRs (S2j); its shares in the events detected from the held-out decode and outside them and the gain at well-sampled places are for the text) and over the events detected from the held-out decode (restriction_heldout_*) or the canonical events (restriction_canonical_non_swr_* and restriction_canonical_swr_*); the *_found_again and *_with_canonical values say how those two sets of events agree. The held-out test split by SWR (the values whose keys start heldout_swr_ or heldout_non_swr_) was made by the tracked scripts/event_evidence/split_state_evidence_by_swr.py at commit e78273b5 (the run's provenance file, read 2026-11-11); the retired claims table called these values pending.",
    ),
    Question(
        "2c.4",
        "Where do event MAPs sit relative to the GPI peak, against matched quiet windows?",
    ),
    Question(
        "2c.6",
        "Are close events produced by the extent of place fields?",
        status="not computed",
        reason="needs a re-decode with a wider Local kernel (a GPU subset)",
    ),
    Question(
        "2d.1",
        "Where does content sit relative to the animal and to the reward wells, SWR against non-SWR?",
        status="partly computed",
        reason="the own-well share waits for the wider-Local-kernel re-decode (question 2c.6)",
        figures=("3e", "3f"),
    ),
    Question(
        "2d.2",
        "How does replay speed differ between SWR and non-SWR events?",
        figures=("3d",),
    ),
    Question(
        "2d.3",
        "How does spatial extent differ between SWR and non-SWR events?",
        figures=("3c",),
    ),
    Question(
        "2d.4",
        "How much does content move within an event (pauses, largest step, reversals), SWR against non-SWR?",
    ),
    Question(
        "2d.5",
        "How concentrated is the posterior (peak, HPD coverage, multimodality), SWR against non-SWR?",
        figures=("S2f",),
        note="Figure 2's supplement, panel S2f, draws the distribution of the events' HPD coverage by class, with the share of events with a tight posterior (the hpd_tight_* values); the pooled class medians (hpd_median_*) are for the text.",
    ),
    Question(
        "2d.6",
        "How does event duration differ between SWR and non-SWR events?",
        figures=("3a",),
    ),
    Question(
        "2d.8",
        "How do theta z-score and speed compare across SWR events, non-SWR events and running?",
        figures=("2g", "S3", "decoder_validation"),
        note="Was question 2c.1: the state the events occur in, not evidence that their content is real.",
    ),
    Question(
        "2d.9",
        "How do per-event spike count, MUA z-score and active tetrodes compare across non-SWR events, SWR events and matched immobile non-event windows?",
        figures=("2", "2h", "S2g"),
        note="Was question 2c.2: events are detected from spiking, so more spikes than a quiet window is not evidence of spatial content.",
    ),
    Question(
        "2d.10",
        "On average, how much of an SWR or non-SWR event is Continuous, and how much Fragmented?",
        figures=("3b",),
    ),
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
        note='The retired claims table\'s "high-value" covers both, and its dropped list calls the CS4 total effect avoidance of recently rewarded arms. Keep the two apart or merge them.',
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
        figures=("5", "5a", "5b", "5c", "5d"),
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
    Question(
        "2a.2",
        "Do held-out likelihood and event-label stability favor the four-state model over the Continuous–Fragmented and reduced models?",
        status="dropped",
        reason="not needed: the paper does not claim the four-state model is better than a reduced one",
    ),
    Question(
        "2a.4",
        "How often does the detector report non-local events on shuffled or thinned real spikes, and in windows where no replay is expected?",
        status="dropped",
        reason="wrongly framed as a false-positive rate: shuffles, thinned spikes and quiet windows are not known negatives; the held-out state test (question 2a.7) and the matched windows of question 2c.3 take its place",
    ),
    Question(
        "2b.6",
        "Does the detector target the represented content rather than an LFP proxy?",
        status="dropped",
        reason="framing with no values: the introduction's argument, not a measurement",
    ),
    Question(
        "2c.5",
        "What spike evidence do matched immobile non-event windows have?",
        status="dropped",
        reason="folded into question 2c.3, as its negative control",
    ),
    Question(
        "2c.7",
        "Do remote events pass the spike-support tests?",
        status="dropped",
        reason="folded into question 2c.3, whose values are given for close and remote events",
    ),
    Question(
        "2c.8",
        "How is the decoded distance from the animal distributed, by SWR status?",
        status="dropped",
        reason="its values are the content's location classes, which answer question 2d.1",
    ),
)
"""Every question, in the outline's order within each section; the dropped
questions last in theirs."""

QUESTION_IDS: dict[str, Question] = {q.id: q for q in QUESTIONS}


def parse_figure(name: str) -> tuple[str, str] | None:
    """The figure id and panel letter ``name`` gives, or None.

    ``"2b"`` is ``("2", "b")``, ``"S2b"`` is ``("S2", "b")``; ``"2"`` and ``"S1"`` have no
    panel (``""``). A name that is no figure of :data:`FIGURES`, or a panel its
    figure does not have, is None.
    """
    if name in FIGURES:
        return name, ""
    figure, panel = name[:-1], name[-1:]
    if figure in FIGURES and panel and panel in FIGURES[figure].panels:
        return figure, panel
    return None


def tag_problems(questions: Iterable[str], figures: Iterable[str]) -> list[str]:
    """What is wrong with a value's tags: an unknown question id, an unknown
    figure, a figure its question does not list, or a question whose status says
    it has no values (:data:`ANSWERED`). Empty when nothing is."""
    questions, figures = tuple(questions), tuple(figures)
    problems = [f"no question {q!r}" for q in questions if q not in QUESTION_IDS]
    known = [f for f in figures if parse_figure(f) is not None]
    problems += [f"no figure {f!r}" for f in figures if f not in known]
    for qid in questions:
        question = QUESTION_IDS.get(qid)
        if question is None:
            continue
        if question.status not in ANSWERED:
            problems.append(f"question {qid} is {question.status}, so has no values")
        for figure in known:
            if figure not in question.figures:
                problems.append(f"question {qid} does not list figure {figure!r}")
    return problems
