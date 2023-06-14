# Finetune
This directory contains the code to fine-tune the open-source LLM with domain specific data.
It uses the adapters like `LoRA`, `QLoRA`, `PEFT`, `SFTT` to reduce the memory overhead.

## Steps to run:
- `pip install virtualenv`
- `virtualenv -p /usr/bin/python venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `export CUDA_VISIBLE_DEVICES=<2,3,4>`  (Replace with your available GPU devices.)
- `python falcon_peft.py`


## Finetune on the custom dataset
- Replace the `dataset_name` (in line #58 of `falcon_peft.py`) with the path to your custom dataset like: `"~/data"`
- It'd contain the data preferably in the `JSON` **Instruction-Following** format like Alpaca dataset.
- For example:
    ```
    [{
        "instruction": "Write a short paragraph about the given topic.",
        "input": "The importance of using renewable energy",
        "output": "The importance of using renewable energy cannot be overstated. Renewable energy sources, such as solar, wind, and hydro power, can be replenished naturally in a short period of time, unlike fossil fuels, which are finite. Utilizing renewable energy reduces greenhouse gas emissions, thus helping to mitigate the effects of climate change. Moreover, it reduces our reliance on non-renewable sources of energy, promoting energy independence, and security. Additionally, renewable energy sources are often cheaper in the long run and can create job opportunities in the green energy sector. Therefore, transitioning to renewable energy is crucial for the health of the planet and the sustainability of our future."
    },
    ]
    ```