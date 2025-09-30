# FDC Estimation Results Archive

This repository contains the results of an exercise in the estimation of streamflow variability for a large sample of hydrologically diverse catchments in and around British Columbia, Canada.

Each station folder contains an html file with a summary of the results.

## Search Results

<div class="search-container">
  <input type="text" id="stationSearch" placeholder="Search by Official ID, name, or source..." onkeyup="filterStations()">
    <div id="searchResults" class="search-results"></div>
  </div>

### HYSETS Dataset

The active and historical monitoring stations examined in this dataset are a subset of the HYSETS dataset {cite}`arsenault2020comprehensive`.  


### Meteorological Inputs

The meteorological data used in the LSTM model is derved from Daymet {cite}`thornton2022daymet`. The data is available at a 1km resolution and includes the following variables:

* `tmax`: Maximum daily temperature [°C]
* `tmin`: Minimum daily temperature [°C]
* `prcp`: Daily precipitation [mm]
* `srad`: Daily shortwave radiation [W/m²]
* `vp`: Daily vapor pressure [Pa]
* `swe`: Daily snow water equivalent [mm]

The data were pre-processed to yield catchment-averaged daily timeseries.  The Processed Catchment-Averaged Meteorological Forcings from Daymet for Streamflow Monitored Catchments in British Columbia, and Transboundary Basins {cite}`kovacek2025metforcings` are available at [https://doi.org/10.5683/SP3/65FXAS](https://doi.org/10.5683/SP3/65FXAS). 


Derived variables include:
* `tmean`: Mean daily temperature [°C]
* `pet`: Potential evapotranspiration [mm/day] (computed using the Penman-Monteith equation)


## Data Dictionary
| Source Code | Description |
|-------------|-------------|
| HYDAT | HYDAT database from Water Survey of Canada (WSC) |
| USGS | United States Geological Survey |
## References
1. Arsenault, R., Brissette, F., Martel, J.-L., Troin, M., Lévesque, G., Davidson-Chaput, J., Gonzalez, M. C., Ameli, A., and Poulin, A.: A comprehensive, multisource database for hydrometeorological modeling of 14,425 North American watersheds, Scientific Data, 7, 243, [https://doi.org/10.1038/s41597-020-00583-2](https://doi.org/10.1038/s41597-020-00583-2), 2020.
2. Thornton, P. E., et al. Daymet: daily surface weather data on a 1-km grid for North America, version 3. ORNL DAAC, Oak Ridge, Tennessee, USA. USDA-NASS, 2019. 2017 Census of Agriculture, Summary and State Data, Geographic Area Series, Part 51, AC-17-A 51 (2016).