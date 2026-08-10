# Paper guide

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
the no-overlap domain and scholarship-improving fairness when housing constraints are absent.

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
reported differences and tests are paired. Standard errors are Monte Carlo standard errors
conditional on the estimated utility parameters.

## Main findings

For 2024, average scholarship-target compliance rises from 68.21% under PCS to 87.84% under
SF-CDA, and residence-target compliance rises from 98.37% to 99.88%. The mean number of
candidates with justified envy falls from 174,801 to 1,028; for scholarship holders it falls
from 78,788 to about 2. Total matches rise by 272, from 510,722 to 510,994, while matched
scholarship holders rise by 9,796. Results for 2022 and 2023 follow the same pattern.

All values and reported Monte Carlo errors are tabulated in
[`../results/paper/mechanism_comparison.csv`](../results/paper/mechanism_comparison.csv).

## Interpretation boundaries

- The exercises are model-based policy comparisons, not causal estimates.
- The code consumes the observed PCS `call_rank`; it does not reconstruct the administrative
  procedure that generated the fixed call order.
- SF-CDA guarantees stage fairness on the general overlapping-trait domain, not unrestricted
  final market stability.
- The paper's exact estimation sample and coefficients require restricted files that are not
  distributed in this repository.
