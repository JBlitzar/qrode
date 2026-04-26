# qrode

https://github.com/user-attachments/assets/d08dce38-8143-4fd8-98fe-3e1ec7e32a56

Every frame is scannable! Try it yourself.

## How this works

TODO write this up in a nicer way

other people have done qr codes for img data but they mostly just abuse payload size and then let ECC do whatever ([here](https://www.reddit.com/r/ProgrammerHumor/comments/kt1zz8/i_created_the_worlds_first_scannable_qr_gif/
))

other people also just heavily abuse ecc and directly blit the image but keep the center pixel in a 3x3 module, like in [this repo](https://github.com/x-hw/amazing-qr)

Qrode one is different: It directly generates payloads that, when encoded, produce qr codes that look like the target image.

### The actual algorithm

TODO expand upon this

Hill climbing with annealing and a bit of image-space perturbation. 

For video that is temporally continuous, you seed subsequent frames with the nearest previous match.

## Run it yourself

```bash
cargo run --release -- --target-kind frames --frames-dir res/frames --mode quality --version 10 --ecc L --payload-mode url --out-png outba/out_badapple.png
```
