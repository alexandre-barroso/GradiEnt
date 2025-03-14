# GradiEnt: A Phonological Analysis Model for Continuous Variables

## Overview

This repository contains a suite of Python scripts implementing the GradiEnt methodology, an approach to analyzing continuous phonetic variables using Optimality Theory frameworks. GradiEnt (Gradient + Entropy) aims to bridge the gap between discrete phonological categories and continuous phonetic phenomena by providing a mathematical framework for analyzing continuous variables without discretization. Currently under development as a research project, by Alexandre M. Barroso @ IEL/UNICAMP, 2024-2025.

### Theoretical Foundations

The methodology is built on several key theoretical principles:

1. **Continuous Constraint Evaluation**: Rather than using discrete violations as in traditional Optimality Theory, GradiEnt implements continuous constraint functions that can evaluate any point in the acoustic space.

2. **Probability Density Functions**: Instead of discrete candidates, phonological categories are represented as probability density functions (PDFs) estimated from empirical data using kernel density estimation (KDE).

3. **Perceptual and Articulatory Restrictions**: The system implements two main types of constraints:
   - Perceptual Restrictions (RP): Model phonological fidelity through sigmoid functions
   - Articulatory Restrictions (RA): Account for articulatory effort and coarticulation effects

4. **Maximum Entropy Optimization**: Uses differential entropy and Kullback-Leibler divergence to optimize constraint weights, producing a MaxEnt distribution that models the phonological system.

### Implementation Architecture

The system consists of two main components:

1. **Core Analysis Engine** (`app.py`):
   - Implements the mathematical framework for GradiEnt analysis
   - Handles data preprocessing and normalization
   - Calculates constraint violations and harmonies
   - Performs optimization of constraint weights

2. **Graphical Interface** (`ui.py`):
   - Provides an accessible interface for linguists
   - Allows parameter configuration for both constraints
   - Supports real-time monitoring of analysis
   - Handles file I/O and visualization display

### Features

- **Continuous Analysis**: Analyzes F1/F2 formant data without discretization
- **Constraint-Based Grammar**: Implements continuous versions of Optimality Theory constraints
- **Maximum Entropy Framework**: Uses differential entropy for continuous probabilistic modeling
- **Statistical Validation**: Includes KL divergence calculations for model evaluation
- **Parameter Optimization**: Implements L-BFGS algorithm for constraint weight optimization

### Mathematical Framework

The system implements several key mathematical components:

1. **Perceptual Restrictions (RP)**:

$R_P(F1) = L - \frac{L^2}{(1 + e^{k_1(F1-F1_{threshold_1})})(1 + e^{-k_2(F1-F1_{threshold_2})})}$

2. **Articulatory Restrictions (RA)**:

$R_A(F1,F2) = \sqrt{(F1-A_{F1})^2 + (F2-A_{F2})^2} \cdot \frac{E_{realized}}{E_{expected}}$

3. **Violation Calculation**:

$v_i = \int e^{R_i(x)-\hat{f}(x)} dx$

4. **Harmonic Score**:

$H = \sum_{i=1}^n p_i v_i$

Where:
- $L$: Maximum violation ceiling
- $k_1, k_2$: Sigmoid curve steepness parameters
- $F1_{threshold_1}, F1_{threshold_2}$: Category transition points
- $A_{F1}, A_{F2}$: Target articulation coordinates
- $E_{realized}$: Articulatory effort for the actual production, computed as:
  
  $E_{realized} = \sqrt{(F1-N_{F1})^2 + (F2-N_{F2})^2}$
  
- $E_{expected}$: Expected articulatory effort, computed as:
  
  $E_{expected} = \sqrt{(A_{F1}-N_{F1})^2 + (A_{F2}-N_{F2})^2}$
  
- $p_i$: Constraint weights optimized through L-BFGS-B
- $v_i$: Constraint violations computed through numerical integration
- $N_{F1}, N_{F2}$: Neutral articulation coordinates
- $\hat{f}(x)$: Kernel density estimation of the empirical distribution

The optimization process minimizes the Kullback-Leibler divergence between the empirical distribution and the MaxEnt distribution:

$KL(p||q) = \int_{-\infty}^{\infty} p \log(\frac{p}{q}) dx$

Where $p$ is the empirical distribution and $q$ is the MaxEnt distribution derived from the constraints.

## Dependencies

- NumPy
- Pandas
- SciPy
- Matplotlib
- scienceplots
- PIL (Python Imaging Library)
- Tkinter
- sklearn (scikit-learn)
- tqdm

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
1. Clone the repository:
```bash
git clone https://github.com/alexandre-barroso/GradiEnt.git
cd GradiEnt
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install numpy pandas scipy matplotlib pillow scienceplots scikit-learn tqdm sympy
```

### Data Preparation

1. Create a text file (e.g., `dados.txt` as provided in the repository) with your formant data
2. Format requirements:
   - Space-separated text file
   - Must include the following columns in order:
     - `Falante` (Speaker ID): numerical identifier for each speaker
     - `Vogal` (Vowel): vowel being analyzed
     - `F1`: First formant frequency values
     - `F2`: Second formant frequency values
   
Example format:
```
Falante Vogal F1 F2
1 e 424.911 1523.522
1 e 440.760 1391.502
2 o 453.980 906.189
```

Requirements:
- Header row is mandatory
- Column names must match exactly
- Values must be separated by spaces
- No missing values allowed in any column
- Decimal points (not commas) for numeric values
- F1/F2 values should be in Hertz

3. Place your data file in the same directory as the scripts

### Running the Analysis

1. Launch the GUI:
```python
python ui.py
```

2. In the GUI:
   - Use "Selecionar Arquivo" to choose your data file
   - Set "Falantes" to specify which speakers to analyze (e.g., "1,3,5" or "0" for all)
   - Set "Vogal" to specify which vowel(s) to analyze (e.g., "e" for [e])
   - Adjust other parameters as needed
   - Click "Iniciar" to run the analysis

## Output Files

The analysis generates several output files:

- `relatorio.txt`: Detailed analysis report
- `amostra_valores.txt`: KDE sample values
- `valores_MaxEnt.txt`: MaxEnt distribution values

## Technical Details

### Parameter Types

1. **Data Parameters**
   - F1/F2 ranges (a_F1, b_F1, a_F2, b_F2)
   - Threshold values (limiar_1, limiar_2)
   - Target values (alvo_F1, alvo_F2)
   - Neutral values (neutro_F1, neutro_F2)

2. **Weight Parameters**
   - lambda_zero: Normalization weight
   - lambda_RP: Perceptual restriction weight
   - lambda_RA: Articulatory restriction weight

3. **Configuration Parameters**
   - Resolution settings
   - File paths
   - Category filters

### Optimization

The suite includes an optimization framework that can automatically determine optimal weight parameters by minimizing the Kullback-Leibler divergence between the empirical and MaxEnt distributions.

## Notes

- The current version of this script is designed for analyzing bivariate distributions with specific focus on F1/F2 parameters
- Supports both manual parameter configuration and automated optimization
- Provides detailed progress feedback during analysis

  GradiEnt is an academic project developed at the State University of Campinas (UNICAMP), Institute of Language Studies (IEL), by Alexandre Menezes Barroso.

## License

This project is licensed under the GNU General Public License v3.0. Feel free to change, improve, distribute, etc, but don't forget to credit me!
