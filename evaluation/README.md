# Prompt inventories by experiment

These files separate surviving local prompt configurations from newly reconstructed candidates. They are not a claim that every image or metric reported in the paper can currently be reproduced from this repository. Reference images, masks, model weights, generated benchmark outputs, and complete metric scripts are not included here.

| File | Scope | Provenance and current limit |
| --- | --- | --- |
| `single_subject_local_prompts.json` | 6 DreamBench single-subject examples and 15 separate showcase prompts | Transcribed from surviving local PA single-subject scripts, not an ARIA multi-subject run. These are examples, not a fixed single-subject benchmark or a completed evaluation manifest. |
| `two_subject_reconstructed.json` | 25 test pairs × 5 prompts, plus 6 proposed held-out pairs × 5 prompts | The first 2 prompts per test pair are verbatim from the surviving local configuration. The other 3 are newly authored: 5 scene descriptions are inferred from a visible qualitative figure, while the rest are proposed coverage prompts. All 6 held-out combinations and their prompts are new proposals, **not** the unknown historical validation set. Each prompt has a `provenance` field. |
| `three_subject_local_config.json` | 12 triples × 5 prompts and bounding-box layouts | Exported from a surviving local rebuttal configuration. Not independently checked against every original output. The configuration refers to `cat3`, whose reference directory was not present in the available local DreamBench data. |
| `four_subject_local_config.json` | 6 scenes × 5 prompts, four seeds, two named methods | Exported from a surviving local configuration. Raw generated outputs were not available for an image-level audit. The configuration refers to `sneaker`, while the available local data has `colorful_sneaker`; resolve this mapping before running. |

The two-subject reconstructed set is a **prospective evaluation plan**. Do not use its generated images to backfill the paper's historical results, or describe its proposed held-out pairs as the set used to select the paper's CPD scale. The `dog_clock` entry additionally carries a warning: its surviving pair configuration uses a corgi-like `dog3`, whereas the qualitative figure shows a fluffy white dog.

For a future run, record the exact prompt, subject-to-reference mapping, seed, model/checkpoint version, generation settings, output path, and whether the image was edited. Keep raw generations separate from presentation-only edits. Report completed and missing pair/prompt/seed cells before aggregating metrics. No images were generated or selected as part of this inventory release.
