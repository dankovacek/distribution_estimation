# Flow Distribution Estimation in Ungauged Basins

Code and notebooks for comparing (log-normal) parametric, k-nearest neighbour, and neural-network methods for daily streamflow distribution estimation (e.g., flow duration curves) in ungauged basins.

The book can be viewed on the web at [https://dankovacek.github.io/distribution_estimation](https://dankovacek.github.io/distribution_estimation)

Release version for the associated paper:
[![DOI](https://zenodo.org/badge/900988443.svg)](https://doi.org/10.5281/zenodo.17756085)


## Edit and Publish


To build the book locally:

```
>jupyter-book build distribution_estimation/
```

To publish to Github Pages:

```
>ghp-import -n -p -f _build/html
```

For more information, see the [jupyter book documentation](https://jupyterbook.org/en/stable/start/publish.html).

## License

This code is licensed under the CC BY 4.0 license. See the LICENSE file for details.
