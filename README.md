# LeNet Study

A paper-to-code study of LeNet-5 from LeCun, Bottou, Bengio and Haffner, 1998, *Gradient-Based Learning Applied to Document Recognition*. Work in progress: local modern-optimizer results are measured below, but the historical optimizer failed and publication/live-demo inspection are unfinished.

## What is reproduced

32x32 input, background -0.1 / foreground 1.175; 6x28x28 C1, trainable sum-pooling S2, sparse Table-I C3, trainable S4, C5 120, F6 84. Scaled tanh 1.7159*tanh(2a/3). C3 uses 60 feature-map links and 1,516 parameters. Total trainable parameters: exactly 60,000. Fixed 84-dimensional +/-1 digit templates with squared Euclidean RBF output, choosing the minimum distance.

Training uses the paper's equation (8) fixed-RBF MSE criterion. The paper also describes the discriminative equation (9); this run does not substitute it silently. A per-example diagonal Gauss-Newton Levenberg-Marquardt update implements Appendix C, including ignoring shared-connection cross terms. Curvature is estimated from 500 training examples per pass, mu=0.02, original 20-pass learning-rate schedule. Training order is randomized once and reused. Weights initialized uniformly +/-2.4/fan-in, biases zero. Sum-pool coefficient starts at .25, an explicit choice because the historical exact initial values are unavailable.

## Fidelity boundary

The output bitmap templates are reconstructed by sampling the author-hosted paper's printed Figure 3. They are an approximation, not recovered original digital RBF weights. All templates are published for inspection. The 1998 paper does not ship every implementation detail or exact random seed; seed 37 and modern PyTorch CPU kernels are used here. This is an architecture/algorithm reproduction with disclosed codebook and implementation differences, not a bit-identical historical rerun or a reproduction of the paper's full document-reading system.

Paper baseline: 0.95% MNIST test error without distortions, 0.8% with distortions. The latter used 540,000 affine-distorted samples in addition to 60,000 originals, but kept the total training session at 1.2M presentations. This reference run uses no distortions. Results must be compared to 0.95%, not presented as matching the 0.8% augmented result. One seed does not establish statistical superiority.

## Extension

A matched modern optimizer protocol compares fixed RBF/MSE with a learned linear/cross-entropy output package. Same initialization seed for shared features, data, batch order, Adam lr .001, batch128, ten epochs, no augmentation. This changes the output and loss together, so it is not an isolated causal head-only effect. Neither modern run is the historical optimizer reproduction. Training schedule is fixed before test evaluation; test results do not select epochs or hyperparameters.

## Sources and rights

Author page: https://leon.bottou.org/papers/lecun-98h
Author-hosted paper: https://leon.bottou.org/publications/pdf/ieee-1998.pdf
Architecture and codebook: pp.7-9; results/data: pp.10-12; initialization and optimizer: pp.40-42.
MNIST: Yann LeCun and Corinna Cortes; CC BY-SA3.0, derivative of NIST data. License statement: https://keras.io/api/datasets/mnist/ . Download mirror and provenance: https://learn.microsoft.com/en-us/azure/open-datasets/dataset-mnist . Exact four compressed-file SHA256 hashes are retained in docs/data-checksums.json. No claim of ownership over the source paper or dataset.


## Measured modern ablation, seed37

| Protocol | Test errors / 10,000 | Error rate |
|---|---:|---:|
| Paper, original optimizer, no distortions | reported 95 | 0.95% |
| This study, fixed RBF/MSE, modern Adam | 122 | 1.22% |
| This study, learned linear/CE, modern Adam | 163 | 1.63% |
| This study, historical-style optimizer attempt | Not evaluated | Failed: saturation |

The modern RBF run has 27 more errors than the paper's reported result, a 0.27 percentage point gap, with implementation/optimizer differences as above. A single seed does not establish a reliable advantage over the linear/CE package. Neither modern run reproduces the original optimizer. No accuracy claim is made for the failed checkpoint. Its first online training MSE was 251.53, with complete C3 saturation and deeper-layer curvature close to zero, so further compute on that checkpoint was stopped.

## Local reproduction and checks

Python dependencies are in research/requirements.txt. `python research/prepare.py` downloads the four named dataset files and records their hashes. `python research/ablation.py` trains the fixed ten-epoch matched modern experiment; existing completed results are skipped so remove/copy result files only in your own working copy when intentionally rerunning. `python research/export.py` exports the completed RBF weights to browser shards. `python tests/check_model.py` checks parameter/link counts, manual gradients vs autograd and nonnegative diagonal curvature. `node tests/parity.mjs` verifies shipped JS inference against ten PyTorch fixtures. The historical trainer is deliberately paused while failed-run.json is present. Do not remove that guard without investigating the documented failure.

Browser inference is plain JavaScript using the published trained weights, with no external runtime or API. Model downloads have a 15-second timeout; Clear cancels pending requests and prevents stale results. Input is normalized to a 20-pixel box and centered by center of mass, padded to32x32. Hand-drawn canvas input is not the MNIST distribution. Results are squared distances, not probabilities. Feature-map tiles show four actual channels per spatial layer, independently normalized for display.
