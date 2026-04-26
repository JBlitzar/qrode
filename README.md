# qrode

https://github.com/user-attachments/assets/d08dce38-8143-4fd8-98fe-3e1ec7e32a56
> <sub> *bad apple played on scannable, protocol-compliant qr codes* </sub>

Every frame is scannable! Try it yourself.

## A brief history of qr code image steg


One strategy is to just heavily abuse ecc and directly blit the image but keep the center pixel in a 3x3 module, like in [this repo](https://github.com/x-hw/amazing-qr)

Another is by using diffusion models to generate an image following a prompt while simultaneously conditioning it to follow a provided qr code image, like in [this huggingface page](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster). Still, this involves post-processing in image-space 

Lastly, there's the strategy to mostly just abuse payload size, directly construct the data regions, and then let ECC do whatever (like [here](https://www.reddit.com/r/ProgrammerHumor/comments/kt1zz8/i_created_the_worlds_first_scannable_qr_gif/
)). This one's admirable because it at least constructs URLs rather than doing any image-space fiddling. It also produces pixel-perfect image results. It's still a little jarring to see the error correction region filled with random noise.

In my opinion, all of these techniques are a little bit *too* good. The qr codes they generate don't look like qr codes (which I guess is the point, they're supposed to look like the target image). 

Qrode is different: It directly generates payloads (with an optional fixed prefix) that, when encoded, produce qr codes that look like the target image.

## How this works

TODO expand upon this

Hill climbing with annealing and a bit of image-space perturbation. I try direct construction and try a bunch of stuff out beyond that. I cache/precompute as much as possible.

I use ssim as a metric over mse matching, which has the nice property of being brightness-invariant (for the most part) but preserving shapes relative to each other. Plus, I get to say that I optimize a perceptual metric.

It's multithreaded and can try multiple paths at the same time, but empirically I've found that it converges to a near-optimal solution pretty quick, and the time investment required to pursue the long tail to get one more percentage point of matching simply isn't worth it.

For video that is temporally continuous, you seed subsequent frames with the nearest previous match.

There are a bunch of CLI options, run with `--help` or code-dig [src/cli.rs](src/cli.rs) to figure out what they do.

## Run it yourself

```bash
cargo run --release -- --target-kind frames --frames-dir res/frames --mode quality --version 10 --ecc L --payload-mode url --out-png outba/out_badapple.png
```
