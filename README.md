# metaImpro

## Description

A repository for improvising with the metapiano.
This creates an UI that allows you to customize some parameters.

## Instalation

This code works with python 3.12.

```bash
pip install -r requirements.txt
```

## Usage
To improvise with a midi file of your choice, make sure it is monophonic and then put it in the folder corpus.
If you want to add a soundfont, modify with the path of the soundfont you added in dpg_interface.

To launch the UI :
```bash
python3 dpg_interface.py
```