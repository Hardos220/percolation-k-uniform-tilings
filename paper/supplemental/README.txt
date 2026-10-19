Supplemental Material: site and bond percolation thresholds of all k-uniform tilings with k <= 10

thresholds.csv - one row per tiling (23,396 rows)
  k              uniformity (number of vertex orbits)
  id             identifier: k, vertex-type orbit structure and catalog file name
  catalog_file   path of the .tes file in the catalog https://github.com/zenorogue/tes-catalog
                 (directory files/tessellations/k-uniform/, commit 6772c7c6)
  composition    vertex types (cyclic sequence of polygon sizes) with their fractions of vertices
  n_v, n_e       vertices and edges per unit cell
  mean_degree    mean vertex degree 2 n_e / n_v
  cell_area      area of the unit cell (unit edge length)
  site, site_err site percolation threshold and its standard error (statistical; for k <= 4 the error
                 of the finite-size extrapolation, not rescaled by chi^2)
  bond, bond_err bond percolation threshold and its standard error
  method         'core: 3 tori, FSS fit' (k <= 4) or 'atlas: 1 torus, n round(s)' (k >= 5; the atlas values
                 carry no finite-size correction and their errors are statistical only)
  tori_site, tori_bond
                 per torus: Lx x Ly unit cells : finite-size estimate (error) : number of samples :
                 aspect ratio of the torus (shorter side / longer side, from Lx|T1| and Ly|T2|)
  chi2_site, chi2_bond
                 chi^2 of the three-torus fit (one degree of freedom), k <= 4 only

cells.jsonl - one JSON object per line, same order as thresholds.csv
  id, k, T1, T2        translation vectors (unit edge length)
  pos                  positions of the n_v vertices of the unit cell
  edges                [u, w, a, b]: edge from vertex u of cell (0,0) to vertex w of cell (a,b),
                       i.e. to the point pos[w] + a*T1 + b*T2
  vertex_configs       vertex type of each vertex (sizes of the polygons around it, in cyclic order)

slip_pairs.csv - pairs with the same unit-cell content and T2' = T2 +- T1/2 (k <= 6)
  id_a, id_b, composition, n_v, d_site, err_site, d_bond, err_bond   (d = p_c(b) - p_c(a))

composition_groups.csv - tilings with identical vertex-type composition (groups of two or more)
  composition, n_tilings, ids (separated by '|'), spread_site, spread_bond (max - min threshold)

atlas_check.csv - finite-size check of the atlas on a stratified sample of 100 tilings (200 thresholds)
  stratum        E = most elongated atlas tori, L = largest unit cells, R = random (5 per k)
  N1, p1, err1   atlas torus: number of sites (bonds), estimate from the atlas runs plus an extension run
  N2, p2, err2   near-square torus with 3.7-4.6 times as many sites
  delta, err_delta   p2 - p1 and its error (measures the finite-size bias of the atlas value)

Errors are one standard deviation.
