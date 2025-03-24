
# Revisiting Financial Sentiment Analysis: A Language Model Approach

corresponding code to the paper: https://arxiv.org/abs/2502.14897

## Easy access
- Neptune AI results for language model experiments can be found [here](https://app.neptune.ai/o/Financial-NLP/org/market-aware-embedding/runs/table?viewId=standard-view&lbViewUnpacked=true&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&groupBy=%5B%22sys%2Fgroup_tags%22%5D&groupByFieldType=%5B%22stringSet%22%5D&groupByFieldAggregationMode=%5B%22auto%22%5D) including tables, confusion matrixes and more.
- the main notebook on Kaggle called tweet-classification can be found [here](https://github.com/hamidm21/Revisit_FSA/blob/master/notebook/tweet-classification.ipynb)
- the final implementation and optimization of Triple Barrier Labeling can be found in the notebook [next_day_prediction](https://github.com/hamidm21/Revisit_FSA/blob/master/notebook/next_day_prediction.ipynb)

  
<img src="https://github.com/user-attachments/assets/effe54e0-2419-411b-bd8c-b65891690d1d" alt="Triple Barrier Labeling" width="300" height="300">

- backtesting experiments and results can be found [here](https://github.com/hamidm21/Revisit_FSA/blob/master/notebook/backtest.ipynb)

## How to run Experiments
use poetry to install the packages with ```poetry install```. for more information go to poetry [docs](https://python-poetry.org/docs/basic-usage/)

then run with ```python src/run.py [Experiment ID]```
## Project Folders and structure
Here are the folders and what they contain:
- raw: unprocessed data
- dataset: processed data
- notebook: notebooks
- src: contains the source code for experiments

## Overall Architecture
![Overall Scheme](https://github.com/user-attachments/assets/e5f85f76-f3f9-42a0-95b8-37bbfdd46026)

## Summary of the Backtesting results
![Backtest Table](https://github.com/user-attachments/assets/78613498-679e-481c-b21c-ff43bcad4e88)
![image](https://github.com/user-attachments/assets/4688e8fd-597d-4bbe-9a98-8ee70c29b665)

