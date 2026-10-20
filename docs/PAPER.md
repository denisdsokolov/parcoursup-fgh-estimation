# Paper guide

*Stable Choice with Overlapping Targets: Theory and Evidence from French College Admissions*

## Research question

The paper studies institutional choice with capacity and exactly two overlapping traits. Under
**one-to-all accounting**, a selected candidate contributes to every target for which they
qualify. Parcoursup provides the application: programs may have minimum target shares for
scholarship holders (`b`) and local residents (`r`), and candidates may have both traits.

## Theoretical result

Pairwise stability combines non-wastefulness with the absence of a one-for-one replacement
that lexicographically improves target deficits and then merit priority. Except for two hard
upper bounds, the bound combinations reduce to a two-soft-lower-bound analogue. Every stable
set is indexed by an **anchor**, the lowest-priority selected dual-trait candidate. Conditional
on the anchor, a unique greedy no-overlap problem remains. For two ordered soft lower bounds,
stable anchors form an interval and all stable sets can be enumerated in quadratic time.

Because the induced choice rules need not satisfy substitutes, a fully stable market-wide
matching need not exist. SF-CDA is therefore deliberately stage-fair rather than a claim of
general full stability. At every stage it keeps previously held candidates and applies the
anchor rule with `k_left` to the residual newcomer problem. The paper proves full stability on
the no-overlap domain and Top-Trait Fairness (hence Top-Trait Stability) in the regular-program
model. The application uses percentage targets, while the general abstract theory uses numerical
lower/upper targets. Non-wastefulness makes the stage-share and integer-threshold SF-CDA
implementations exactly equivalent. All programs choose before candidates respond.
See [TARGETS.md](TARGETS.md) for the residual targets and proof-to-code correspondence.

## Empirical design

The main 2024 administrative market contains 886,585 candidates, including 153,081 scholarship
holders, and 21,414 programs with 523,160 places; 522,435 candidates are observed matched.
Boarding programs are treated as ordinary programs in the simulations, as specified in the
paper (they represent 3.6% of programs and 6.2% of capacity).

Applicant utilities are estimated using the feasible-choice-set approach of Fack, Grenet, and
He (FGH). A program enters a candidate's personalized set when the candidate's observed call
rank is above the final program cutoff, with the outside option added. The 2024 estimation
sample has 88 strata, 197 programs, and 28,312 candidates. Alternative-specific constants for
non-sampled programs are imputed from program capacity, targets, boarding status, demand, and
program-category indicators.

The counterfactual uses 1,000 preference draws. Each draw is used for both PCS and SF-CDA, so
reported differences are paired. Parentheses in the mechanism-comparison tables are
**across-simulation standard deviations**, conditional on the estimated utility parameters.
Preference-coefficient tables instead report estimation standard errors.

## Main findings

For 2024, SF-CDA matches **9,796 additional scholarship holders** (100,329 versus 90,533,
+10.8%), while total matches rise by 272 (510,994 versus 510,722). Among scholarship holders,
22,037 prefer SF-CDA and 2,903 prefer PCS, a net ordinal balance of 19,134. The corresponding
overall balance is 299, reflecting 28,396 SF-CDA-preferring and 28,097 PCS-preferring candidates.

Final-enrollment scholarship-target fulfillment rises from 68.21% under PCS to 87.84% under
SF-CDA; residence-target fulfillment rises from 98.37% to 99.88%. Mean justified-envy counts
fall from 174,801 to 1,028; for scholarship holders, from 78,788 to about 2. In 2022 and 2023,
additional scholarship matches number 7,722 and 7,335. See the [public paper tables](../results/paper/)
for all means and standard deviations.

## Interpretation boundaries

- The exercises are model-based policy comparisons, not causal estimates.
- The code consumes the observed PCS `call_rank`; it does not reconstruct the administrative
  procedure that generated the fixed call order.
- SF-CDA guarantees stage fairness on the general overlapping-trait domain, not unrestricted
  final market stability.
- The paper's exact estimation sample and coefficients require restricted files that are not
  distributed in this repository.

The included `estimate_fgh.py` specification is the 2024 baseline. Exact annual re-estimation
requires the full original annual specification and saved sample. The selected coefficients
printed in the paper do not identify all terms of an annual estimation; the simulator applies
the known annual interactions present in the supplied `beta_hat.csv`.
