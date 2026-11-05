# Classification settings

Classification used the private datasets from Monkeys B and H. These datas are not included in this release.

The settings below follow the manuscript. Classification and regression use the same convolutional backbone, 64-dimensional time-conditioned session descriptor, and channel-wise modulation network. The only architectural difference is the output layer: three logits for classification and two velocity outputs for regression.

## Task and protocol

- Three-class decoding for Monkeys B and H.
- Target calibration budget in the main experiment: five trials per class.
- NAV inputs were formed from MUA using a 300 ms analysis window and 50 ms bins. The resulting input dimension was 480 for Monkey B and 396 for Monkey H.
- Monkey B used 200 ms before to 100 ms after the Center Release event; Monkey H used 100 ms before to 200 ms after the Target Hit event.
- Original valid-channel inputs were used without channel padding.
- LODO folds: 14 for Monkey B and 8 for Monkey H.
- Forward-chain folds: 9 for Monkey B and 3 for Monkey H.
- Forward chaining uses only sessions recorded before the target day and requires at least five source sessions.
- Chance accuracy is 33.33%.

## Shared model

- Three one-dimensional convolution blocks with kernel sizes 7, 3, and 3; stride 1; padding 1; and 32, 64, and 128 output channels.
- The session descriptor is the sum of a learnable session offset and a linear projection of the normalized recording-day position.
- Each convolution output is modulated by a channel-wise scale and bias computed from the session descriptor.
- Dropout is 0.2 after the flattened convolutional representation.
- The shared fully connected representation has 256 units.
- The classification output layer has three logits and uses cross-entropy loss.

## Training

- Stage 1: SGD, learning rate `1e-3`, momentum 0.9, batch size 36, 10 epochs.
- Stage 2: 50 meta-iterations, 3 inner-loop steps, inner learning rate 0.01, AdamW outer optimizer with learning rate `1e-4`, weight decay `1e-4`, anchor coefficient 0.05, and gradient clipping 2.0.
- Stage 3: SGD, learning rate `1e-2`, momentum 0.9, batch size 32, at most 100 epochs, and early-stopping patience 10.
- Ten seed bases are used; the seed for fold `f` is `seed_base + f`.
