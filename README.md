<div align="center">

<h1> Style-Transfer: Apply the style of an image to another image </h1>

[![Website](https://img.shields.io/badge/Website-ClarityAI.cc-blueviolet)](https://ClarityAI.cc) [![Twitter Follow](https://img.shields.io/twitter/follow/philz1337x?style=social)](https://twitter.com/philz1337x)

[![Replicate](https://img.shields.io/badge/Demo-Replicate-purple)](https://replicate.com/philz1337x/style-transfer)
![GitHub stars](https://img.shields.io/github/stars/philz1337x/style-transfer?style=social&label=Star)

My post on X/Twitter: https://x.com/philz1337x/status/1771559668910858246?s=20

</div>

# 👋 Hello

I build open source AI apps. To finance my work i also build paid versions of my code. But feel free to use the free code. I post features and new projects on https://twitter.com/philz1337x

# 🚀 Options to use Style-Transfer

## User friendly Website

If you are not fimilar with the tools described here, you can use my paid version at [ClarityAI.cc](https://ClarityAI.cc)

## Deploy and run with cog (locally or cloud)

If you are not familiar with cog read: <a href=https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md>cog docs</a>

- clone the repo

```bash
git clone github.com/philz1337x/style-transfer
cd style-transfer
```

- download the weights

```bash
pip install diffusers
python download_weights.py
```

- start a prediction

```bash
cog predict -i image="link-to-image" -i image_style="link-to-style-image"
```

## Replicate API for app integration

- go to https://replicate.com/philz1337x/style-transfer/api

## Run with A1111 webUI

Use ControlNet Canny for the structure and IPAdapter for the style.

## Run als a single script

- change the image links in dev.py
- run dev.py

```bash
python dev.py
```
