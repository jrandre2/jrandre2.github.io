# Standalone benchmark reproduction: Hurricane Helene, western North Carolina

Rebuilds the principal Table 2 scores and all plotting data for Figures 1–2 from frozen, source-linked derived inputs. This is a reproducibility bundle for the fixed benchmark, not a new model fit choice, operational validation, or full re-extraction of the archives. No external drive, credentials, geographical libraries, or network are needed once Python dependencies are available.

## Run

Tested with Python 3.12.2. From this directory:

```sh
python -m pip install -r requirements.txt
python -B reproduce.py
shasum -a 256 -c SHA256SUMS
```

A run writes CSVs under `reproduced/`; use `--out /path/to/new/results` to keep the checked-in examples unchanged. The script checks all matching numeric Figure 1–2 values against the frozen originals in `expected/`, and checks timing and NC-trigger rescoring metrics. The tolerance accommodates arithmetic differences across numerical builds. SHA-256 covers frozen inputs, independent expected output and executable files; generated example outputs are deliberately not hashed. Figure map *data* is reproduced, not the PDF or drawing. County IDs are five-character FIPS codes; percentage scores are in percentage points.

## Three separate designs

| Output | Design and source boundary |
|---|---|
| Table 2, Figure 1 data, peak metrics | Refit 589 rows/11 earlier storms with the original storm-weighted log-ridge. The 499 inherited non-NC labels each have one archived eligible snapshot; 90 NC labels use available historical snapshots. Refit from their original count labels and derived county weather features. Score 21 Helene peak labels independently calculated from 420 direct county report records (seven observations per county within the first week); also score first 48 h, stable first week, omit Sept 29, omit Ian, and sparse versus full archive. The matched terrain sample uses 578 complete training rows and modern-vintage county summaries. |
| Timing and trigger metrics | Fully refit the five matched timing arms on 490 records/8 earlier storms; held-out target uses common strictly after-12-hour Helene peak horizon. Separately rescore frozen NC-approach trigger predictions on those full same-window Helene outcomes (not a refit of the NC trigger model, not a reconstruction of its track detector). |
| Table 2, Figure 2 data, recovery sensitivity and leave-one-storm-out errors | Start from 3,458 reconciled county-report snapshots with actual report URLs, then rebuild lag-dependent pairs and train from scratch. The main set has 1,203 pairs/8 pre-Helene storms and 210 test pairs; each alternative uses the same main rows. Other scopes use backup-only or joint backup-and-cause support. Retain each lag’s own fit/availability rules; report per-lag days 3–9 prior/target denominator match and the intersection of 140 test targets at 6 h and 24 h. |

Main MAEs, in table order: **71.663631, 70.797830, 70.161314, 13.768347, 28.984446, 26.724540, 28.895786 pp.** In the shared-140 lag sensitivity, the six-/24-hour persistence MAEs are 7.343789/15.024292 pp; the respective transport MAEs are 12.113804/13.355458 pp. The fits are not held fixed across lag conditions.

## Provenance and limits

See `sources.csv` and the input provenance CSVs for source links and archived issuance metadata. Every FCC outcome row includes an exact official public PDF link; historical recovery source reconciliation is frozen. All count-based fitting/scoring, denominator eligibility, report pairing, log transforms, lag variants, and shared-target restriction above are recalculated. Recovery report times are observation clocks, not assumed posting timestamps; no missing report turns into a zero.

This is **not** raw-to-final reproduction. It does not download, reparse or redistribute FCC PDFs; resolve source-quality conflicts from PDF lines; regenerate the sample inherited from the national archive; derive county overlaps from raw forecast GIS inputs; verify GIS product release to end users; or rebuild terrain rasters, HRRR/MRMS coverage, geographic hazard associations, plot artwork, or the NC-approach track detector. The upstream historical archive extraction and county forecast geography must be independently reconstructed to audit their derivation completely; exact source URLs and derived inputs make those boundaries explicit. No national-directory path is required for this run. Tools use only this folder’s frozen files.

## Additional revision analyses

Run `python -B matched_peak.py` to select the timing-matched single reports from the included 1,075-row eligible observation ledger, fit the matched models, and compare all 29 level-error scores and individual predictions with the saved results. Ranking is checked at nine decimal places so fitted values differing only by floating-point noise remain ties. Run `python -B extended_lags.py` to reconstruct prior/target pairs under the expanded delay grid and check both available-at-each-delay and shared-case comparisons. These scripts write only under `reproduced/` by default. The coverage ledger is frozen from the existing selected archives, and it does not contain all reports ever published.

The folder **Model comparison extracts** adds frozen score files and chronological audit tables for the newer model comparisons. Run `python "Model comparison extracts/build.py"` from this directory to verify the extracts without reconstructing the upstream fits. The broader pilot contains the scripts that generated those fits. The main benchmark scripts and their input formats are unchanged.

## Hazard figure

Run `python -B hazard_associations.py` to regenerate the five county partial rank correlations used in Figure 3 and check them against the plotted values. This script starts from 21 previously derived county observations. It checks the early peak and initial outage columns against the included daily outage shares. It does not recalculate location weighting or overlays from raw geographic inputs. The five point estimates concern two different endpoints. No uncertainty interval or causal conclusion was calculated for the figure.
