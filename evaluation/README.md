# Prompt inventories by experiment

These files separate surviving local prompt configurations from newly reconstructed candidates. They are not a claim that every image or metric reported in the paper can currently be reproduced from this repository. Reference images, masks, model weights, generated benchmark outputs, and complete metric scripts are not included here.

| File | Scope | Provenance and current limit |
| --- | --- | --- |
| `single_subject_local_prompts.json` | 6 DreamBench single-subject examples and 15 separate showcase prompts | Transcribed from surviving local PA single-subject scripts, not an ARIA multi-subject run. These are examples, not a fixed single-subject benchmark or a completed evaluation manifest. |
| `two_subject_reconstructed.json` | 25 test pairs × 5 prompts, plus 6 proposed held-out pairs × 5 prompts | The first 2 prompts per test pair are verbatim from the surviving local configuration. The other 3 are newly authored: 4 scene descriptions are inferred from a visible qualitative figure, while the rest are proposed coverage prompts. All 6 held-out combinations and their prompts are new proposals, **not** the unknown historical validation set. Each prompt has a `provenance` field. |
| `three_subject_local_config.json` | 12 triples × 5 prompts and bounding-box layouts | Exported from the generation configuration matching the 12 locally preserved result folders. Prompt/seed coverage of those outputs has not been independently verified. |
| `four_subject_local_config.json` | 6 scenes × 5 prompts, four seeds, two named methods | Exported from a surviving local configuration. The source's generic `sneaker` label is resolved to the available `colorful_sneaker` reference directory for a future rerun, with the mapping recorded in `reference_resolution`; the original four-subject outputs were not available to confirm which sneaker was historically used. |

The two-subject reconstructed set is a **prospective evaluation plan**. Do not use its generated images to backfill the paper's historical results, or describe its proposed held-out pairs as the set used to select the paper's CPD scale. The `dog_clock` prompts follow the surviving `dog3` pair configuration rather than the different dog visible in the qualitative figure.

For a future run, record the exact prompt, subject-to-reference mapping, seed, model/checkpoint version, generation settings, output path, and whether the image was edited. Keep raw generations separate from presentation-only edits. Report completed and missing pair/prompt/seed cells before aggregating metrics. No images were generated or selected as part of this inventory release.
