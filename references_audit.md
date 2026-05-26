# Reference Audit

Checked on 2026-05-26.

## Safe Corrections Applied

- `APA:83`: corrected title from `Publications Manual` to `Publication Manual of the American Psychological Association`.
- `Aho:72`: expanded title to the canonical volume title.
- `lee2023rlaif`: corrected title from `RLAIF vs. RLHF: ...` to `RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback`.
- `guan2025rstar`: changed from `@inproceedings` / ICML 2025 to `@misc` arXiv because I found the arXiv record but did not find a stable ICML/PMLR proceedings page.
- `DeepSeek-R1`, `VinePPO`, `R1-Zero-Like`, `Automated Process Supervision`, and `Beyond Markovian` kept as arXiv entries with arXiv URLs/DOIs.
- ACL Anthology entries now include ACL URLs; these may be redundant if the paper already imports `anthology.bib`.
- OpenReview entries now include OpenReview URLs for ICLR/TMLR/NeurIPS-track records.
- Common NeurIPS/PMLR/AAAI/JMLR/Foundations-and-Trends entries now include stable venue URLs or DOI links.

## Not Found / Needs Caution

- No fully nonexistent reference was found among the pasted entries.
- `guan2025rstar` as an ICML 2025 proceedings paper was not verified; use the arXiv citation in `references_verified.bib`.
- `zhang2024rest` is listed as NeurIPS 2024 with an OpenReview URL. If you need camera-ready NeurIPS proceedings page metadata, re-check once the exact proceedings page is required.
- Long-author papers using `and others` are valid BibTeX but not maximally complete. If the submission style requires full author lists, expand these entries from the linked official pages.

## Optional Cleanup

- The first template references (`Aho:72`, `APA:83`, `Chandra:81`, `andrew2007scalable`, `Gusfield:97`, `rasooli-tetrault-2015`, `Ando2005`) look like ACL template leftovers unless they are actually cited in the manuscript. Remove unused entries before final submission.
- Papers already contained in ACL Anthology can be removed from this file if `anthology.bib` is included, to avoid duplicate keys/records.
