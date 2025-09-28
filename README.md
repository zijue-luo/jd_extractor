# jd_extractor
Extract Skillsets in Job Descriptions using LLM

## Installation
### Create Conda Environment
```
conda create -n jdextractor python=3.11 -y
conda activate jdextractor
```

### Install Dependencies
```
pip install -r requirements.txt
```
### configure API Keys to Environment
create `.env` file, follows format of `.env.example`

## Run
```
conda activate jdextractor
streamlit run app_web.py
```

## Debug
remember to add `-m` to debug single files, e.g.
```
python -m core.prompt_builder
```