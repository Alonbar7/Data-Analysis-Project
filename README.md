# SISE2601 Project data description
Team 12 - Alon Bar, Amir Belinsky, Tal Ber and Nitzan Graf

# Repository Description
- Proposal: The first draft of the project, same as submitted.
- project: The final project folder containing the Rmd report, full analysis for sciCode benchmark and full analysis for all benchmarks.
- Data: All of the datasets used in the project.

# Main Data Description
Main Data File: llm_price_performance_tracker_2026-03-31.csv

Data Link: https://www.kaggle.com/datasets/kanchana1990/llm-price-performance-tracker-march-2026

Total Observations (Rows): 453

Total Variables (Columns): 8 / 34 (Variables kept)

- model_name (Text): The official commercial or research name of the model.
- mmlu_pro (Continuous Numerical): Massive Multitask Language Understanding Professional score.
- gpqa_diamond (Continuous Numerical): Google-Proof Q&A benchmark score, representing advanced expertise in physics, biology, and chemistry.
- humanitys_last_exam (Continuous Numerical): Performance on a highly complex benchmark designed to test the absolute frontier of AI knowledge.
- livecodebench (Continuous Numerical): Performance on real-world, dynamic programming and algorithmic challenges.
- scicode (Continuous Numerical): Evaluation of the model's ability to solve scientific coding problems and numerical methods.
- is_open_source (Binary): Indicates whether the model's weights are publicly accessible (TRUE) or restricted (FALSE).
- blended_cost_usd_per_1m (Continuous Numerical): A weighted average cost per 1 million tokens, simulating standard usage patterns.
- pricing_tier (Categorical): Free (0), Budget (<0.50), Mid (0.50–5), Premium (5–30), Ultra (>30), Unknown (no data)

# Secondary Data Description

Second Data File: LifeArchitect_Models.csv

Data Link: https://lifearchitect.ai/models-table/

Total Observations (Rows): 886

Total Variables (Columns): 3 / 47 (Variables kept)

- model_name (Text): The official commercial or research name of the model.
- Params (Continuous Numerical): The parameter count of the model.
- Tokens trained (B) (Continuous Numerical): The amount of token used to train the model.
- Arch (Categorical): The Architecture of the model (MoE, Dense)
