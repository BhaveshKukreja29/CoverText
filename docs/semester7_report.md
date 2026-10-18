# Linguistic Steganography and Steganalysis in Small Language Models

Semester seven report. The numbers in the results section are copied from `docs/phase1/sweep.json`, produced by `python scripts/reproduce.py --seed 0 --n-samples 4 --n-contexts 8 --device cpu`.

## 1. Introduction

The goal of this project is to hide an arbitrary bit string in text from a small open language model, recover those bits exactly, and measure how easily a detector can tell the result from ordinary text. The model is Qwen2.5-0.5B-Instruct. The cover prompts and the clean corpus both come from WikiText-103. The secret is a random bit string, not a sentence.

Two encoders are compared. Arithmetic coding steers each token so that the hidden bits pick a sub-interval of the model's next-token distribution. Minimum-entropy coupling, in the form used here, assigns tokens to disjoint bins and lets an arithmetic coder choose the bin, which keeps the token marginal equal to the model's truncated distribution. Two detectors are compared. One scores how surprising the text is under the same model. The other trains a linear classifier on the model's last-token hidden state.

The quantity we care about is the tradeoff between capacity, in bits per token, and detection, as area under the ROC curve. Decode accuracy and the perplexity gap between stego text and its prompt are reported beside that curve.

## 2. Literature review

Simmons' prisoner's problem is the setting. Alice sends Bob a text that a warden also reads. The prompt is the shared context. The warden sees the continuation, not the prompt. Cachin's definition of security asks whether the warden's view of stego text is distinguishable from the view of text drawn from the cover distribution. A scheme whose token marginal matches the language model has zero KL divergence from that model at each step, which is the strongest form of that requirement for a next-token sampler.

Ziegler, Deng, and Rush (EMNLP 2019) encode bits by arithmetic coding over a language model's next-token distribution. The bits name a point in the unit interval. Each token claims a sub-interval of length equal to its probability, so the chance of emitting a token is exactly that probability when the bits are uniform. The implementation here follows that construction, with a 32-bit streaming coder and a top-50 truncation, and it drops candidate tokens that the tokenizer cannot round-trip. That last filter is there because a subword encoding that does not invert breaks the bit mapping even when the arithmetic is correct (the failure mode discussed by Nozaki and Murawaki, ACL 2022).

Schroeder de Witt et al. (ICLR 2023) construct perfectly secure steganography by coupling the model's token distribution to a uniform message so that the token marginal is unchanged. An exact uniform binning is not available for a neural softmax: the set of distributions that partition into exact dyadic masses has measure zero, and splitting a token across bins makes decoding ambiguous. The encoder in this project therefore keeps tokens whole. It packs the top-50 tokens into disjoint bins, codes the bin with the arithmetic coder under the bin masses, and samples inside the chosen bin with the renormalized token probabilities. The marginal of a token is then the bin mass times the conditional probability inside the bin, which multiplies back to the original probability.

On the detector side, a perplexity or distribution-shift score is the usual baseline: stego steering spends some probability mass on less likely tokens, so the average surprise goes up. Wang et al. (arXiv:2406.04218) argue that a stronger detector is a classifier on a language model's own features. Their classification mode runs the text through the model once and maps the final representation to a cover-versus-stego probability with a new linear layer, instead of asking the model to generate a verdict token by token. The classifier in this project is that classification mode on a frozen Qwen backbone. The paper also fine-tunes the backbone with LoRA. We do not, because the encoders share this model and a trained backbone would change the distribution they encode against.

## 3. Methodology

### 3.1 Data and payloads

WikiText-103 is split by the Hugging Face split names, which are fixed. Paragraphs are kept when they contain between 64 and 256 tokens under the Qwen tokenizer. The encoder prompts are the first eight training paragraphs after a deterministic shuffle with seed 0. The clean class is the first eight test paragraphs. Those two lists are disjoint by construction, and the detector's training half is split from its evaluation half with the same seed, so no evaluation paragraph is used to calibrate or to fit.

Payloads are independent fair bit strings of length 8, 16, 32, and 64. Each sample has its own seed derived from the sweep seed, the payload size, and the sample index. Four samples are drawn at each payload size.

### 3.2 Encoders

Both encoders condition on the prompt, then generate only the continuation. A trailing space is inserted when the prompt does not already end in whitespace, so a byte-pair merge cannot join the prompt to the first stego token. At each step the model returns the top 50 next-token probabilities at temperature 1. Tokens that do not survive `decode` then `encode` are removed and the remaining mass is renormalized. Generation stops when the coder has absorbed the payload, or raises if 100 tokens are not enough.

Arithmetic coding consumes the payload directly against those probabilities. The coupling encoder first packs the tokens into `2^floor(log2(k))` bins by always adding the next token to the currently lightest bin. The coder selects a bin under the bin masses. A token is then drawn inside the bin. Decoding reads the bin off the observed token and feeds that bin index back to the coder. Because no token belongs to two bins, the bin is unambiguous.

### 3.3 Detectors

The perplexity detector reports two statistics and a one-sided score. Sequence perplexity is the exponential of the mean negative log probability of the observed tokens. The second statistic is the mean conditional entropy deficit, `mean(-log P(x_t) - H(P_t))`. Under the model's own sampling distribution this deficit has expectation zero. The score maps only upward deviations from a calibration mean, computed on the training half of the clean paragraphs, into `[0, 1]`. Text that is more predictable than that mean scores as clean.

The classifier takes the last-layer hidden state at the last non-padding token. In a causal model that state is the only vector that has attended to the whole string. If the text is longer than 256 tokens, the suffix is kept, because a generative payload is written at the end. Features are z-scored with the training-set mean and standard deviation. A two-class linear layer is trained by minimizing mean cross-entropy with AdamW. The head is reinitialized and trained separately at each payload size. The backbone stays frozen, and the forward pass is chunked so the activation memory does not grow with the corpus. The score is the softmax probability of the stego class.

### 3.4 Metrics

Capacity is payload bits divided by the number of continuation tokens. Decode accuracy is the fraction of bits that match, with a length mismatch counted against the longer string, and the fraction of payloads that match exactly. Fluency loss is the perplexity of the continuation minus the perplexity of the prompt. A continuation of one token has no conditional probability, so its perplexity is undefined and that sample is omitted from the fluency mean. `n_fluency` records how many samples entered the mean. Detector performance is AUROC with clean labeled 0 and stego labeled 1. Higher scores are treated as more stego.

### 3.5 What this comparison does not isolate

The clean texts are WikiText paragraphs of 64 to 256 tokens. The stego texts are continuations, often only a few tokens for an 8-bit payload and a few dozen for 64 bits. A detector can therefore use length. That is a real confound. It is also the comparison the phase-one corpus specifies: held-out paragraphs against encoder output. A length-matched control would draw a natural continuation of the same length from the same prompt and score that against the stego continuation. This report does not claim the AUROC numbers below are free of the length effect.

## 4. Results

The sweep has not been filled in yet.
