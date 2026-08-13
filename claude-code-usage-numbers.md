# Claude Code usage numbers

*Analysis date: 13 August 2026.* A point-in-time snapshot, not a maintained document —
every figure below describes the session as it stood on that date and will drift the
moment work continues. Nothing in the codebase depends on these numbers, and no test
holds them honest, which is why the date is at the top rather than in a footnote.

`grow-up` was written end to end by [Claude Code](https://claude.com/claude-code) in a
single conversation, supervised by the repository owner. Every commit in the repository
except the initial one carries the same session id, so the scope here is unambiguous: it
is the whole project.

## What the human typed

| | |
|---|---|
| Prompts | **47** |
| Words across all prompts | **5,244** |
| Median prompt | 30 words |
| Mean prompt | 112 words |
| Shortest / longest | 4 / 2,798 words |
| Quartiles (p25 / p75) | 16 / 63 words |
| Prompts of 10 words or fewer | 6 |
| Interruptions of a tool mid-flight | 24 |

The mean is the least useful number here: one 2,798-word prompt drags it to four times
the median. Half of all steering was done in 30 words or fewer.

## What the model did

| | |
|---|---|
| Assistant messages | 1,564 |
| Tool calls | 989 |
| Words written back to the human | 21,496 |

| Tool | Calls |
|---|---|
| Bash | 419 |
| Edit | 291 |
| TaskUpdate | 71 |
| Write | 64 |
| TaskCreate | 38 |
| Read | 28 |
| ExitPlanMode | 23 |
| ToolSearch | 13 |

## What came out

| | |
|---|---|
| Commits | 35 |
| Lines added / removed | 12,757 / 908 |
| Source | 5,480 lines across 16 modules |
| Tests | 5,186 lines across 21 files, **630 tests** |
| Documentation | 1,015 lines (README, CLAUDE.md, config example) |
| Wall clock, first commit to 1.3.0 | 3 days, 2 hours |

Releases: **1.0.0** (26 commits), **1.1.0** (3), **1.2.0** (2), **1.3.0** (3).

## Ratios

- One word typed produced **4.1 words of reply** and **0.19 tool calls**.
- Test code and source code are within 6% of the same size.
- Roughly **21 tool calls per prompt**.

## Method, and what these numbers are not

Counts come from the Claude Code session transcript on disk (3,600 records) and from
`git`, not from recollection — the model's own context had been summarized several times
over three days, so anything counted from memory would have been a guess.

Three caveats worth stating plainly:

- **"Words typed" is what reached the model as prompt text.** Pasted logs, plans and
  error output are included, and account for most of that 2,798-word maximum. It is not
  a measure of typing effort.
- **Interruptions are counted as events, not as prompts.** They carry no text of their
  own; the 47 prompts are separate from the 24 interruptions.
- **None of this measures quality.** 630 tests is a count of tests, not evidence that the
  timelapse looks right. Everything in this repository that depends on real photographs
  was verified by the repository owner — the model never had access to the Immich
  instance or its API keys.
