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
        [
            {
                "instruction": "Give three tips for staying healthy.",
                "input": "",
                "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
            },
            {
                "instruction": "Write a short paragraph about the given topic.",
                "input": "The importance of using renewable energy",
                "output": "The importance of using renewable energy cannot be overstated. Renewable energy sources, such as solar, wind, and hydro power, can be replenished naturally in a short period of time, unlike fossil fuels, which are finite. Utilizing renewable energy reduces greenhouse gas emissions, thus helping to mitigate the effects of climate change. Moreover, it reduces our reliance on non-renewable sources of energy, promoting energy independence, and security. Additionally, renewable energy sources are often cheaper in the long run and can create job opportunities in the green energy sector. Therefore, transitioning to renewable energy is crucial for the health of the planet and the sustainability of our future."
            },
        ]
        ```
    - `input` is the context to the instruction. It can be empty as well.
