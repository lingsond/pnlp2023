<div align="center">

# UniWue - Praktikum NLP SS2023 - Multilingual Evaluation

[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-390/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![lightning-hydra-template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

<!---
[![Paper](http://img.shields.io/badge/paper-arxiv.wannabe-B31B1B.svg)]()
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)]()
--->
</div>

## Description

Evaluation (and fine-tuning) of some LLMs on several tasks (NLI, QA, NER, etc.).

The project was supposed to be a 3-person project, however, the other two didn't participate until the end of the project.


## Contributors
- Dirk Wangsadirdja - Task: NLI, Dataset: XNLI
- ~~Nada Aboudeshish~~
- ~~Zihao Lu~~


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/lingsond/pnlp2023.git
cd pnlp2023

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/lingsond/pnlp2023.git
cd pnlp2023

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

# For XNLI Related Information
## Important Folders
- /experiments/ - where the experiments raw results are kept.
- /results/ - where the spreadsheets for the experiment results are kept.
- /templates/ - for the jinja templates
- /yaml/ - configuration files to run the containers on university cluster (kubernetes).
- /lingson/configs/ - configuration files for processing XNLI dataset (best prompt selection, standard evaluation, peft training, peft evaluation).
- /lingson/bigscience/ - PEFT models (generated from the peft fine-tuning process).

## Important Scripts and Python Files
### Best Prompt Selection
- /lingson/pipeline_best_prompt_newest.py
### Standard Evaluation
- /lingson/pipeline_evaluate.py
### PEFT Training
- /lingson/pipeline_train_peft.py
### PEFT Evaluation
- /lingson/pipeline_evaluate_peft.py

## Notes
General process:
- Select best prompt template
- Standard evaluation with pre-trained models
- Fine-tuning models with PEFT methods
- Evaluation again using PEFT fine-tuned models

After each process, the author might have think/find a way to improve the code, and during these steps, some file structures might have changed. 

And since after finishing each process step, the author didn't need to repeat/redo the previous process, no checking were done to ensure the scripts integrity. 

So, it is possible that the pipelines for best prompt selection and standard evaluation are not working anymore because of the changes.

## How to run

Each of the pipeline scripts can be called with the config file as parameter. All configurations needed are coded inside the configuration files.

```bash
# For the processes involving XNLI dataset 
python lingson/pipeline_scripts.py --config configs/config_file.json
```

# For Other Datasets
The other contributors were supposed to work on these.