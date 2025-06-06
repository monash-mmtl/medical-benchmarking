# MEDICAL BENCHMARKING

This repository contains a multi-turn medical benchmarking project inspired by ideas and implementations from [CRAFT-MD](https://doi.org/10.1038/s41591-024-03328-5) and [AgentClinic](https://doi.org/10.48550/arXiv.2405.07960).

We include a series of generated cases for evaluation from differential diagnoses and presenting complaints taken from Murtagh's General Practice textbook.

Cases are tagged according to the 
## Data Generation Workflow
- `main_generation.py`: First generation run of `all_cases.jsonl`. 
  - Outputs: X_presenting_complaint folder: contains individual cases of X complaint, presenting_complaint_all_cases.jsonl: contains all cases appended of X complaint, failed_cases.json: contains cases failed to generate in initial run

- `retry_failed_dfiferentials.py`: Retries all failed cases from `failed_cases.json`.
  - Outputs: regenerates failed cases with multiple retries, and adds generated files to folders and relevant all_cases.jsonl files.
 

### Clinic workflow
Accepts X_presenting_complaint_all_cases.jsonl as input for evaluation runs
