# 🛰️ Cloud and Land Classification Map Generator

This script generates a classification map using two NetCDF files:
1. A **cloud mask file** (e.g., `cma_extended`)
2. A **land/physiography file** (e.g., `landuse`)

The output is a color-coded image showing clouds, land, water, snow/ice, and no-data areas, with optional subsetting and customization.

---

## 🔧 Features

- Accepts cloud and land NetCDF files as input
- Optional region subsetting (x/y range)
- Option to classify **all land** (regardless of cloud cover)
- Saves classification map as PNG to an output folder
