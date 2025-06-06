# MEDICAL BENCHMARKING

This repository contains a multi-turn medical benchmarking project inspired by ideas and implementations from [CRAFT-MD](https://doi.org/10.1038/s41591-024-03328-5) and [AgentClinic](https://doi.org/10.48550/arXiv.2405.07960).

We captured 100 differential diagnoses and related presenting complaints from Murtagh's General Practice textbook for case generation.

All presenting complaints have a number of differential diagnoses which are captured under the subheadings of 'Probability Diagnosis', 'Serious Disorders' and 'Pitfalls' accoridng to the textbook.
- These are stored in `tables_list/diagnoses.jsonl` and fetched in the workflow.

We generated all presenting complaints relevant captured under the 100 differential diagnoses (total ~3000 cases) and present them here in `all cases.zip`.
- Cases are tagged with the relevant subheadings for subgroup analysis.

## Data Generation Workflow
- `main_generation.py`: First generation run of `all_cases.jsonl`.
  - Fetches presenting complaints and differential diagnoses from tables_list/diagnoses.jsonl for generation.
    - As of last run of 06/25, cases are generated using *gemini-2.5-pro-preview-03-25* using Vertex AI SDK.
    - requires a `service-account.json` for credentials.
  - Outputs: X_presenting_complaint folder: contains individual cases of X complaint, presenting_complaint_all_cases.jsonl: contains all cases appended of X complaint, failed_cases.json: contains cases failed to generate in initial run.

- `retry_failed_dfiferentials.py`: Retries all failed cases from `failed_cases.json`.
  - Outputs: regenerates failed cases with multiple retries, and adds generated files to folders and relevant all_cases.jsonl files.
 

### Clinic workflow
Accepts X_presenting_complaint_all_cases.jsonl as input for evaluation runs.
