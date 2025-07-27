# Analysis Scripts

This directory contains all the analysis scripts for the 1D density slope analysis project.

## Core Analysis Files

### `density_slope_analyzer.jl`
The main analysis module containing all core functions:
- `analyze_slopes_from_file()` - Analyze slopes from saved state files
- `calculate_density_slope()` - Linear regression on density regions
- `plot_density_with_slopes()` - Visualization with fitted slopes
- `plot_density_simple()` - Simple density profile visualization

### `analyze_size_scaling.jl`
Comprehensive size scaling analysis script that:
- Processes all files in `../dummy_states/for_current_analysis`
- Fits multiple models: `a/L²`, `b/L`, `a1/L + a2/L²`, `b/(L+a)`, `b/(L²+a)`
- Performs log-log power law analysis
- Generates comparison plots with all fits
- **Usage**: `julia analyze_size_scaling.jl`

## Test and Example Scripts

### `test_slope_analyzer.jl`
Comprehensive test script for the density slope analyzer:
- Tests analysis on individual files
- Creates visualization plots
- Validates fit quality and asymmetry detection
- **Usage**: `julia test_slope_analyzer.jl`

### `example_slope_analysis.jl`
Example usage demonstrations:
- Single file analysis examples
- Multiple file comparison
- Quick analysis functions
- CSV export functionality
- **Usage**: `julia example_slope_analysis.jl`

### `quick_slope_analysis.jl`
Lightweight script for quick slope analysis tasks.

## Key Features

### Exclusion Logic
All analysis functions exclude middle-1, middle, and middle+1 points when fitting to avoid boundary effects in the periodic density profiles.

### Periodic Boundary Treatment
The combined periodic fit treats the right region as continuous with the left through the boundary, significantly improving R² values.

### Multiple Model Fitting
The size scaling analysis fits 5 different models and compares their performance:
1. **a/L²** - Simple inverse square scaling
2. **b/L** - Simple inverse scaling  
3. **a1/L + a2/L²** - Polynomial combination (often best fit)
4. **b/(L+a)** - Offset inverse scaling
5. **b/(L²+a)** - Offset inverse square scaling

### Power Law Analysis
Log-log analysis extracts scaling exponents directly from the data with error estimates.

## Generated Outputs

### Plots
- `slope_vs_system_size_all_fits.png` - Main comparison plot with all model fits
- `slope_vs_size_log_log.png` - Log-log power law analysis
- `slope_vs_size_log_log_linear_scale.png` - Power law fit on linear scale
- Individual density profile plots with naming: `density_slope_analysis_L_{L}_g_{gamma}.png`

### Data Analysis
- Summary tables of slopes vs system size
- Model comparison with R² values  
- Power law scaling parameters with error estimates
- Scaling relationship verification (1/L, 1/L² tests)

## Usage Examples

```julia
# Run complete size scaling analysis
julia analyze_size_scaling.jl

# Test individual file analysis  
julia test_slope_analyzer.jl

# See example usage patterns
julia example_slope_analysis.jl
```

## Path Structure
All scripts expect to be run from the `analysis/` directory and reference data files using relative paths:
- `../dummy_states/for_current_analysis/` - Input data files
- `../saved_states/` - Additional test data files
- Current directory - Output plots and analysis results
