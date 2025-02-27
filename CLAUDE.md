# VWYO Project Reference

## Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate vwyo
```

## Commands
- **Run a test file**: `python test_file.py` (e.g., `python test_1.py`)
- **Run Streamlit visualization**: `streamlit run test_squarecb.py` or `streamlit run test_epsilon.py`

## Code Style
- **Imports**: vowpalwabbit first, followed by standard library, then data science libraries
- **Formatting**: 4-space indentation, docstrings with triple quotes
- **Naming**: snake_case for functions and variables
- **Error handling**: Use ValueError for invalid inputs
- **Constants**: ALL_CAPS for constants (e.g., USER_LIKED_LAYOUT)

## Project Structure
- Test files prefixed with `test_`
- Models saved with `.model` extension
- Uses VowpalWabbit for contextual bandit algorithms
- Visualizations with matplotlib, plotly, and streamlit