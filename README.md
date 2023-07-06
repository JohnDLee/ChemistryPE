# Chemistry Prompt Engineering for LEF reaction prediction
### by John Lee, Taicheng Guo, Xiangliang Zhang

This research project attempts to utilize prompt engineering to harness the power of LLM for chemistry reaction prediction problems. We limit the scope to only include linear electron flow (LEF) reactions, since the electron flow of such reactions intuitively seems to be decomposable in to a CoT problem.

We compare our results to ELECTRON(insert link).


## To Do

- [x] Retrieve LEF USTPO data used in ELECTRO
- [x] Be able to parse data into products, reactants, reagents
- [x] Evaluate zero-shot and few-shot baselines
- [x] Be able to decompose strings in to electron flow retrieve intermediate SMILES strings
- [] Test LLM capabilities in predicting next step in a reaction
    - [] inital remove
    - [] remove or add
    - [] middle steps
- [] Evaluate methods for CoT.