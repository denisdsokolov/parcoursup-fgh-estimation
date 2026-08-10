# Residency histograms

These files contain aggregate summaries of program-specific residency status.

For each year there is a `size_i` by `size_j` matrix. Cell `(i, j)` contains the number of
students whose:

1. share of wishes with residency lies in `[i / size_i, (i + 1) / size_i)` (the final bin
   includes 1); and
2. number of wishes equals `j` (the final bin also includes larger values).

The calculations use approximately 880,000 candidates and 21,000 programs per year. Source
tables: `a_adm`, `a_rec_grp`, `a_rec`, `a_voe`, `c_can_grp`, `g_can`, `g_tri_ins`, `i_ins`,
and `sp_g_tri_aff`.
