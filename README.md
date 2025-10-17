# metaImpro

## Description

Repository for real-time improvisation on a keyboard using various machine-learning models (Factor Oracle, Markov, etc.) with a flexible UI.

## Features
- 🎹 Models

    - Factor Oracle

    - Markov model

    - (and more coming soon)

- ⚙️ Customizable parameters via a DearPyGui interface

- 🎶 Corpus selection from a corpus/ folder

## Authors

[Florent DB](https://github.com/FlorentDB)

## Dependencies

- This projects requieres python 3.12
- Dependencies listed in requirements.txt

## Requirements
 - Python 3.12+

 - Dependencies listed in requirements.txt
 
 - [fluuydsynth](https://github.com/FluidSynth/fluidsynth/wiki/Download)

## Instalation

1. Clone the repository

```bash
git clone https://github.com/scrime-u-bordeaux/metaImpro.git
cd metaImpro
```

2. Create a venv

```bash
python3.12 -m venv venv
source venv/bin/activate 
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4. Prepare your corpus

Create a corpus/ folder at the root (it already exists with some examples, add your monophonic .mid files to it)

## Usage

Start the interface:

```bash
python3 dpg_interface.py
```

Then, in the window:

1. Choose your MIDI device

2. Select a MIDI file from the corpus/

3. Choose your template and its parameters

4. Click Start Impro


## Common errors

If you get an error during the installation, you might have some reccurent problems as :
you installed the python package fluydsynth instead of pyfluidsynth.

You get this error : ModuleNotFoundError: No module named 'pkg_resources'. It means you may need to reinstall or install your setup-tools.