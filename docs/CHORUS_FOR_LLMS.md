# Chorus for LLMs

This document explains the Chorus Engine and its output files to a language
model. Paste it alongside any Chorus output (consensus Markdown, JSON bundle,
AI context pack, and so on) when asking an LLM to summarise, fact-check, or
otherwise reason about a transcript.

---

## 1. What Chorus does

Chorus Engine is a local, multi-pass consensus transcription tool. Instead of
transcribing an audio file once, it:

1. **Generates several cleaned variants** of the source audio (a high-pass
   filtered pass, a dynamic-range-normalised pass, and a denoised pass), in
   addition to the original recording.
2. **Transcribes each variant independently** using a local OpenAI Whisper
   model.
3. **Aligns the resulting transcripts word by word** and takes a vote at
   every position: the word most variants agree on becomes the canonical
   word for that position, and the fraction of variants that agree on it
   becomes its confidence score.
4. **Optionally reconstructs low-confidence words** using either a local
   spaCy grammatical/semantic pass or a local Ollama LLM pass, before
   rendering the final documents.
5. **Optionally attaches speaker labels** via `pyannote.audio` diarisation.
6. **Renders the result** into a Markdown "consensus" document, a JSON
   bundle, an AI context pack, and (optionally) PDF/DOCX/SRT/VTT exports.

The point of this architecture is redundancy: a word that only one variant
"hears" is far more likely to be a mishearing or hallucination than a word
that three or four variants agree on independently.

---

## 2. Output format reference

All files share a `{stem}` prefix — the sanitised base filename of the
source recording. A typical run produces:

| File | Produced by | Purpose |
|------|-------------|---------|
| `{stem}_consensus.md` | `consensus_merger/renderer.py` | The primary, annotated transcript. Confidence tiers are shown inline via Markdown decorators (see §3). |
| `{stem}_bundle.json` | `export_engine/exporter.py::export_transcript_bundle` | Structured, machine-readable version of the same data: every variant's raw text, the full word-vote sequence, and aggregate statistics. **Prefer this file for programmatic extraction.** |
| `{stem}_ai_context.md` | `export_engine/ai_context.py::generate_ai_context_pack` | A document written specifically to accompany an LLM prompt: methodology, processing configuration, confidence statistics, the clean transcript, an uncertainty table, and usage guidance. Always generated. |
| `{stem}_best_guess.txt` | `export_engine/exporter.py::export_best_guess` | The cleanest possible plain-text transcript: every position resolved to its single highest-agreement word, with **no** brackets, confidence markers, or statistics. Always generated. |
| `{stem}_most_likely.txt` | `export_engine/exporter.py::export_plain_text` (`include_low=True`) | Plain transcript with LOW-confidence words shown as `[word?]`. |
| `{stem}_most_likely_clean.txt` | `export_engine/exporter.py::export_plain_text` (`include_low=False`) | Plain transcript with LOW-confidence words omitted entirely. |
| `{stem}_diarised.md` | `diarisation/diariser.py::render_diarised_md` | Speaker-labelled transcript (only produced when diarisation is enabled). |
| `{stem}_speakers.json` | `diarisation/diariser.py` | Sidecar mapping speaker labels (e.g. `SPEAKER_00`) to human-assigned names, if any were set in the UI. |
| `{stem}_consensus.pdf` / `.docx` | `export_engine/exporter.py` | Formatted renderings of the consensus document with confidence highlighting preserved. |
| `{stem}_consensus.srt` / `.vtt` | `export_engine/exporter.py` | Word-level or segment-level subtitle files, timed against the original Whisper pass. |

If you have been given only one file, `{stem}_bundle.json` and
`{stem}_ai_context.md` are the two most useful for LLM consumption:
the bundle for structured extraction, the context pack for a
prose-and-table summary designed to sit in a prompt.

---

## 3. Confidence tier semantics

Every word in the consensus sequence is assigned to one of three tiers,
based on what fraction of the transcribed variants agreed on it (the
default is four variants: original, high-pass, normalised, denoised):

| Tier | Agreement | Meaning | Rendering in `.md` files |
|------|-----------|---------|---------------------------|
| **HIGH** | ≥ 75 % (configurable via `CONSENSUS_THRESHOLD` in `config.py`, or per-run via the UI sidebar sliders) | The word was heard consistently across variants. Treat as reliable. | Plain text, no decoration. |
| **MEDIUM** | Exactly 2 of 4 variants (a 50 % split) | Real disagreement between variants — worth a second look but usually not noise. | `==word==` (Markdown highlight syntax). |
| **LOW** | Present in only 1 variant | Likely a mishearing, filler artefact, or hallucination introduced by a single audio-cleaning pass. | `**~~word~~**[^NN%: variant / variant]` — bold, struck through, with a footnote naming the observed forms and the percentage. |

**Guidance for an LLM reading a consensus document:**

- Quote HIGH-confidence passages with normal confidence.
- Treat MEDIUM-confidence words as plausible but flag them if the surrounding
  sentence would materially change in meaning depending on the word.
- Treat LOW-confidence words with real scepticism — check whether the
  variants listed in the footnote change the meaning of the sentence, and
  prefer to say "the transcript is uncertain here" over guessing.
- For anything safety-, legally-, or medically-critical, always recommend
  human verification of MEDIUM/LOW passages regardless of how plausible the
  chosen word looks.

---

## 4. Word-vote structure (the consensus algorithm)

Internally (`consensus_merger/alignment.py`, `consensus_merger/sequence_alignment.py`),
Chorus represents the aligned transcript as an ordered list of `WordVote`
records, one per consensus position:

| Field | Type | Meaning |
|-------|------|---------|
| `word` | `str` | The canonical word chosen for this position — the form observed by the largest group of variants (ties are broken by first occurrence). This is the word that appears in every rendered output, including `{stem}_best_guess.txt`. |
| `count` | `int` | How many variants agreed on this word (or a fuzzy match of it). |
| `total` | `int` | How many variants were compared (normally 4). |
| `confidence` | `float` | `count / total`, rounded to 3 decimal places. |
| `tier` | `str` | `"HIGH"`, `"MEDIUM"`, or `"LOW"`, derived from `confidence` and `count` per the thresholds in §3. |
| `variants` | `list[str]` | The distinct word forms actually observed across variants at this position (includes the winning form). |

Two alignment strategies exist, selected via `ALIGNMENT_STRATEGY` in
`config.py` (default: `"sequence"`):

- **`"sequence"`** — a Needleman-Wunsch-style alignment
  (`sequence_alignment.py`) that correctly handles one variant inserting or
  dropping words relative to another.
- **`"positional"`** — a simpler, faster index-based comparison
  (`alignment.py`) that assumes all variants produce a similar word count;
  it is more prone to cascading misalignment after an insertion/deletion.

Fuzzy matching (Levenshtein-based, `SIMILARITY_THRESHOLD = 0.80` by default)
groups near-identical word forms (e.g. minor spelling variants from
different Whisper passes) into the same voting bucket before counting.

`{stem}_bundle.json`'s `"consensus"` array is a direct JSON serialisation of
this `WordVote` list (see §5) — it is the most reliable place to read the
raw vote data from, rather than re-parsing the Markdown decorators.

---

## 5. `bundle.json` schema

Produced by `export_engine/exporter.py::export_transcript_bundle`. Fields,
verified against the current implementation:

```json
{
  "meta": {
    "stem": "recording",
    "source_filename": "recording.m4a",
    "generated_at": "2026-07-11T12:00:00+00:00",
    "chorus_version": "4.0.1",
    "schema_version": 1
  },
  "variants": {
    "original": {
      "text": "full transcript text for this variant",
      "language": "en",
      "model": "base",
      "device": "cpu"
    },
    "highpass": { "...": "..." },
    "normalised": { "...": "..." },
    "denoised": { "...": "..." }
  },
  "consensus": [
    {
      "word": "hello",
      "tier": "HIGH",
      "confidence": 1.0,
      "count": 4,
      "total": 4,
      "variants": ["hello"]
    }
  ],
  "statistics": {
    "total_words": 1,
    "high": 1,
    "medium": 0,
    "low": 0,
    "high_pct": 100.0,
    "medium_pct": 0.0,
    "low_pct": 0.0
  }
}
```

`meta.schema_version` identifies the contract revision of this file (currently
`1`); it is bumped only when a field is renamed or removed, so consumers can
rely on the fields above being present for any bundle with the same
`schema_version`. `meta.chorus_version` records the Chorus release that
produced the bundle.

To extract structured data programmatically:

- **Full transcript** — join `consensus[].word` with spaces, in order.
- **Per-variant raw transcripts** — `variants.<key>.text` for any variant
  key present in `variants`.
- **Reliability score** — `statistics.high_pct`.
- **Every uncertain word and its alternatives** — filter `consensus` for
  entries where `tier != "HIGH"`, and read `variants` on each entry for the
  candidate forms that were considered.

---

## 6. Export formats — metadata and markup conventions

| Format | Module | Notes |
|--------|--------|-------|
| **PDF** | `export_pdf()` (WeasyPrint) | Renders the consensus Markdown with confidence highlighting preserved (yellow = MEDIUM, red strikethrough = LOW). No machine-readable structure beyond the visual styling. |
| **DOCX** | `export_docx()` (python-docx) | Same tier mapping as PDF, expressed as Word highlight colours and strikethrough runs. |
| **SRT** | `export_srt()` | SubRip subtitles. Word-level cues (≤ 6 words per cue) when Whisper word timestamps are available; falls back to segment-level cues otherwise. Confidence markup is stripped — subtitle text is always plain. |
| **VTT** | `export_vtt()` | WebVTT equivalent of the above, `.` instead of `,` as the millisecond separator, and a `WEBVTT` header line. |

None of the subtitle or document exports carry confidence-tier metadata in
machine-readable form — only the consensus Markdown and `bundle.json` do.
If you need to correlate a subtitle cue with its confidence tier, cross
reference the cue text against `bundle.json`'s `consensus` array.

---

## 7. Worked prompts

These examples assume the reader has been given `{stem}_bundle.json` and/or
`{stem}_ai_context.md` alongside the request.

**Summarisation**

> Using the attached Chorus transcript (`bundle.json`), summarise the
> discussion in three bullet points. Only rely on HIGH- and MEDIUM-tier
> words for factual claims; note explicitly if a key detail falls in a
> LOW-confidence span.

**Fact-checking**

> The attached transcript claims `<quote>`. Check the `consensus` array in
> `bundle.json` for the words making up this claim. If any of them are
> tier `"LOW"`, list the alternative `variants` observed and state whether
> a different reading would change the claim.

**Speaker-intent analysis**

> Using the attached `{stem}_diarised.md`, describe each speaker's stance
> on `<topic>`. Where a decisive word falls in a `==highlighted==` (MEDIUM)
> or `**~~struck~~**` (LOW) span in the corresponding `{stem}_consensus.md`,
> flag that the speaker's exact wording is uncertain rather than asserting
> it confidently.

---

## 8. See also

- [`README.md`](../README.md) — installation, usage, and architecture overview.
- [`docs/CONFIGURATION.md`](CONFIGURATION.md) — every configurable threshold
  referenced in this document (`CONSENSUS_THRESHOLD`, `SIMILARITY_THRESHOLD`,
  `ALIGNMENT_STRATEGY`, and more).
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — internal module design.
