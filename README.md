# qrode

https://github.com/user-attachments/assets/d08dce38-8143-4fd8-98fe-3e1ec7e32a56
> <sub> *bad apple played on scannable, protocol-compliant qr codes* </sub>

Every frame is scannable! Try it yourself.

## A brief history of qr code image steg

Strategy 0: Everyone's seen the option on most online qr code generators to just stick a logo in the middle. ECC takes care of the rest, you obliterate some pixels. Let's move on to some actual techniques.

One strategy is to just heavily abuse ecc and directly blit the image but keep the center pixel in a 3x3 module, like in [this repo](https://github.com/x-hw/amazing-qr). This is actually utilizing a deep fact about qr code scanner implementations, which is that the extracted bitmap is more biased to the center of each pixel. So this messes around with the surrounding subpixels, and hopes that ECC fixes up any bodged cells.

Another is by using diffusion models to generate an image following a prompt while simultaneously conditioning it to follow a provided qr code image, like in [this huggingface page](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster). Still, this involves post-processing in image space.

Lastly, there's the strategy to mostly just abuse payload size, directly construct the data regions, and then let ECC do whatever (like [here](https://www.reddit.com/r/ProgrammerHumor/comments/kt1zz8/i_created_the_worlds_first_scannable_qr_gif/
)). This one's admirable because it at least constructs URLs rather than doing any image-space fiddling. It also produces pixel-perfect image results. It's still a little jarring to see the error correction region filled with random noise. To put it differently, the images are almost *too* good, and it makes the finder patterns seem out of place.

In my opinion, all of these techniques are a little bit *too* good. The qr codes they generate don't look like qr codes (which I guess is the point, they're supposed to look like the target image). 

Qrode is different: It directly generates payloads (with an optional fixed prefix) that, when encoded, produce qr codes that look like the target image.

## How this works

TODO expand upon this

Hill climbing with annealing and a bit of image-space perturbation. I try direct construction as a seed and try a bunch of perturbations out beyond that. I cache/precompute as much as possible.

I use ssim as a metric over mse matching, which has the nice property of being brightness-invariant (for the most part) but preserving shapes relative to each other. Plus, I get to say that I optimize a perceptual metric.

It's multithreaded and can try multiple paths at the same time, but empirically I've found that it converges to a near-optimal solution pretty quick, and the time investment required to pursue the long tail to get one more percentage point of matching simply isn't worth it.

For video that is temporally continuous, you seed subsequent frames with the nearest previous match.

All of that for a ~70% pixel match to the target if I'm lucky. For reference, random noise is 50%. I polled a few people by showing them static images with patterns baked in, and most didn't notice anything off until I pointed it out. It's also pretty blurry, you can't see any detail.

Larger QR codes weren't much better.

![](docs/qr_circle.png)

> *Did you see the circle?*

So the video format helps a lot! The pixel-level noise is blurred away perceptually each frame, and the blurry shapes are what's left behind, similar to the premise of [this shader](youtube.com/watch?v=RNhiT-SmR1Q). 

It's possible to respin lack of good convergence as... subtlety, I suppose. At least the whole thing is textured homogenously. Just think! None of the other direct construction methods would work on QR codes this small, because ECC would dominate. Qrode doesn't care about the distinction between ECC and data regions. It's all a black-box optimization problem. Which is simultaneously a smart and a dumb approach. 

In summary:

Upsides: 
  - It looks like a normal QR code

Downsides: 
  - It looks like a normal QR code.



## Run it yourself

Maybe one day I'll publish to crates or build some binaries. 
```bash
cargo run --release -- --help
```

There are a bunch of CLI options, run with `--help` or code-dig [src/cli.rs](src/cli.rs) to figure out what they do.


```bash
cargo run --release -- --target-kind frames --frames-dir res/frames --mode quality --version 10 --ecc L --payload-mode url --out-png outba/out_badapple.png
```
