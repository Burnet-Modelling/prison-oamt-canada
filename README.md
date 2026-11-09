# Canada OAMT model

Code for a stochastic compartmental model of opioid agonist maintenance therapy (OAMT) in Canadian federal prisons. It projects the costs and health consequences of the switch to long-acting injectable buprenorphine (LAI-BPN) as first-line OAMT, against a counterfactual in which OAMT stays at its end-of-2024 pattern. The model is built in [Atomica](https://github.com/atomicateam/atomica).

## Files

- `run_main.py` runs the calibration, the policy change and counterfactual scenarios, the economic outcomes, the figures and the sensitivity analyses
- `utils_Canada.py` model set-up, scenario runs, calibration, outcome and cost calculations, and the output tables
- `utils_plotting.py` figures
- `pnsp_oat_framework_Canada.xlsx` Atomica framework (compartments, parameters, transitions)
- `pnsp_oat_databook_Canada.xlsx` databook (model inputs and calibration targets, simulation starts in 2015)
- `input_pars.xlsx` cost, utility and post-release inputs with their ranges and sources

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
python run_main.py
```

Tables and figures are written to `results/`. `run_main.py` runs everything in one go (calibration, both scenarios, the main analysis and all the sensitivity analyses) with 2,000 simulations per scenario, so a full run takes a while. Set `n_samples` at the top of the script for a quicker test run.
