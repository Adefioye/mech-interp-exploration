PYTHONPATH=. python experiments/train/train.py \
    --sparsity 0.05 \
    --ranks 100,200,300,400 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --model-name gpt2-small \
    --factorization-mode mlp \
    --layers 0,3,6,9,11 \
    --data-path data/final_dataset_20_concepts.json \
    --model-device mps \
    --data-device cpu \
    --fitting-device mps \
    --base-path . \
    --save-path experiments/artifacts/ \
    --seed 42

PYTHONPATH=. python experiments/snmf_interp/generate_concept_context.py \
  --models-dir experiments/artifacts \
  --output-json experiments/artifacts/concept_contexts.json \
  --layers 0,3,6,9,11 \
  --ranks 100,200,300,400 \
  --num-samples-per-factor 25 \
  --context-window 15 \
  --sparsity 0.05 \
  --seed 42 \
  --model-name "gpt2-small" \
  --factor-mode mlp \
  --data-path data/final_dataset_20_concepts.json \
  --model-device mps \
  --data-device cpu

PYTHONPATH=. python experiments/snmf_interp/generate_input_descriptions.py \
  --input-json experiments/artifacts/concept_contexts.json \
  --output-json experiments/artifacts/input_descriptions.json \
  --model openai/gpt-4o-mini \
  --env-var OPENROUTER_API_KEY \
  --layers 0,3,6,9,11 \
  --k-values 100,200,300,400 \
  --top-m 10 \
  --max-tokens 200 \
  --concurrency 50 \
  --retries 5

PYTHONPATH=. python experiments/concept_detection/generate_sentences.py \
  --input-json experiments/artifacts/input_descriptions.json \
  --output-json experiments/artifacts/generated_sentences.json \
  --model openai/gpt-4o-mini \
  --layers 0,3,6,9,11 \
  --k-values 100,200,300,400 \
  --n-per-mode 5 \
  --concurrency 50 \
  --max-tokens 100 \
  --retries 3 \
  --jitter-min-ms 50 \
  --jitter-max-ms 300 \
  --env-var OPENROUTER_API_KEY

PYTHONPATH=. python experiments/concept_detection/benchmark.py \
  --mode mlp \
  --model-name "gpt2-small" \
  --layers 0,3,6,9,11 \
  --k-values 100,200,300,400 \
  --sparsity s0.05 \
  --save-path experiments/artifacts/interp_results.json \
  --concept-data experiments/artifacts/generated_sentences.json \
  --models-root experiments/artifacts \
  --device mps \
  --data-device cpu
