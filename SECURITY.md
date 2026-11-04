# Security boundaries

Static site. No login, trackers, cookies, uploads, external model calls, persistent user data, secrets, payment paths or writable backend. All code, weights and examples are hosted together. Inputs are a bounded 280x280 drawing canvas. DOM output uses textContent and locally created elements. Downloads have a 15-second timeout and cancellation; Clear cancels pending requests and prevents stale results. A failed model fetch shows no result. Inference is deterministic local JavaScript, without dynamic evaluation. Content Security Policy restricts scripts, workers, style and network to this origin, with no objects/forms.

Server authorization, database rules, rate limiting, queues and money idempotency are not applicable because no writable backend exists. Browser caching is provider-controlled. Model distances are not calibrated probabilities. Canvas input differs from the MNIST evaluation set and can be misclassified. Do not treat it as dependable automated document recognition.

Research downloads use a fixed public MNIST mirror; file hashes are recorded. The historical optimizer run failed and is guarded from blind continuation. A public failed-run record is included; checkpoints are research artifacts, not user data. Source paper/dataset copyright and license boundaries remain in README.

Checkpoint loading uses weights_only=True. Early local failed-run checkpoints contained NumPy state and are not shipped or loaded by the guarded trainer. The fixed-source dataset URL is the only download scanner suppression; filenames and HTTPS origin are constants, not user input.
