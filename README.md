# VALUEFLOW

## 1) Environment Setup

```bash
conda create -n vllm python==3.12
conda activate vllm
pip install -r requirements.txt
```

## 2) Generate Responses

- **Single-value case**
  ```bash
  bash scripts/generate_responses_single.sh
  ```

- **Multiple-value case**
  ```bash
  bash scripts/generate_responses_multiple.sh
  ```

## 3) Evaluate Responses

```bash
bash scripts/evaluate_responses.sh
```

> You may change the **model**, **theory**, **value**, **intensity**, and **prompt format** used for generation.

---

## License

### Code and Models
We release all code and pretrained models under the **Apache 2.0** license, permitting broad reuse and extension.

### Value Intensity Database (VIDB)
Because VIDB is derived in part from third-party datasets with heterogeneous terms, we restrict redistribution and use of VIDB to **non-commercial research** only. Users must also honor the original licenses of the underlying datasets.

**Primary sources and licenses**
- **MFRC** — Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Social Chemistry** — Creative Commons Attribution–ShareAlike 4.0 International (**CC BY-SA 4.0**)
- **ValueNet** — Creative Commons Attribution–NonCommercial–ShareAlike (**CC BY-NC-SA**)
- **ValueEval** — Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **ValuePrism** — **AI2 ImpACT License**, Medium Risk Artifacts (“MR Agreement”)

When using VIDB, ensure any downstream distribution, sharing, or publication of text excerpts complies with these original licenses (e.g., attribution, share-alike, and non-commercial clauses where applicable).
