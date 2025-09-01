## 🛰️ Cloud and Land Classification Map Generator

This Python script generates stepwise classification maps from two NetCDF files:

# Cloud mask file (e.g., cma_extended)

# Land/physiography file (e.g., landuse)

The output consists of four PNG images showing:

Base classification (No data, Water, Ice, Clouds, Land)

Classification + Proximity (overlay for areas near clouds or land)

Neighbors – Ice (ice neighbors affecting water classification)

Neighbors – Water (water neighbors affecting ice classification)

It also supports optional subsetting and customization of land classification.

## 🔧 Features

Accepts cloud and land NetCDF files as input

Optional region subsetting (x_min, x_max, y_min, y_max)

Option to classify all land regardless of cloud cover (--all_land)

Automatically saves all four classification maps in a specified output folder (--output)

Implements color-coded maps with clear legends, including proximity overlays and neighbor relationships
