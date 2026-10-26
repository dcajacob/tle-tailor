# Included code, data and notices

Project code retains the MIT [LICENSE](../LICENSE). Adapted python-sgp4/JAX code retains its upstream credits and complete [MIT notice](../licenses/python-sgp4-MIT.txt). [NOTICE](../NOTICE) records retained attribution. gmatpy and other libraries are installed as separate dependencies and carry their own notices; TLE Tailor does not vendor their native trees.

The historical attributed `coarse_fit.py` helper is retained in local research history but is excluded from packages and is not needed by supported runtime paths. No licensing change is made by moving examples. Frozen public comparison files and their notices are retained for reproducible algorithm comparisons.

Historical mission datasets, provider ephemerides, catalogs, BSPs, papers, original research notebook outputs and private mission workflows are not bundled as public sample data. Supply authorized inputs externally when a data-backed workflow needs them. Do not include private data, credential files or generated outputs when sharing your work. Example placeholder identifiers are not assigned spacecraft identities.

The built distributions have an explicit content boundary in `distribution-policy.json`; that boundary is separate from the entire historical source checkout. Required project/upstream licenses and attribution notices remain in the distribution.
