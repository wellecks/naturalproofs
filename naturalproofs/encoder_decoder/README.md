### Autoregressive and Joint Models

We implement the autoregressive model as an `EncoderDecoder` model, and the joint model as a one-step special case with KL-divergence loss.

As a result, the two model implementations share code and are contained in this directory.
- `model.py`: autoregressive model.
- `model_joint.py`: joint model.
- `predict.py`: **generation** and **retrieval** evaluation for the autoregressive and joint models.
- `utils.py`: command-line tokenization (and other utilities)

See the Appendix for further details.
