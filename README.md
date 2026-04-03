AI Photo Renamer
================

The `ai-photo-renamer.py` Python script analyzes photos using AI and renames them to
have a brief description in the filename. It uses [LM Studio](https://lmstudio.ai/) for
the actual AI work, meaning that all processing is done on your local hardware and you
don't need to worry about usage limits or sending private data to potentially
untrustworthy third-parties.

It also supports identification of people (and dogs!) in order to aid the identification
process to make the new filenames more specific.


Requirements
------------

To use `ai-photo-renamer.py`, you'll need the following:

- A Python 3.12+ environment and [`uv`](https://github.com/astral-sh/uv)
- [LM Studio](https://lmstudio.ai), configured as a [local network
  server](https://lmstudio.ai/docs/developer/core/server/settings)
  - A LLM model that supports image identification. By default, the script uses
    [`qwen3-vl-4b`](https://lmstudio.ai/models/qwen/qwen3-vl-4b)


Usage
-----

In the directory that your images are in, create a YAML file named `.analyze.yml`:

```yaml
---
# General description which is passed to the LLM to assist in image identification
description: A trip to New York City
# List of people to identify. Each name should have a corresponding `<name>.jpg` (or
# `<name>.png` image in the people directory.
people:
  - Bob
  - Alice
  - Eve
# Same as the `people` list, each dog should have a corresponding image in the people
# directory.
dogs:
  - Harry
  - Sally
```

All fields are optional, and strictly speaking, creation of the YAML file is optional as
well, but it will assist in creating better image descriptions. When running the script,
you'll need to pass in the path to the images to analyze, as well as an optional
`--people-dir` which is used to identify people/dogs.

The script will not actually rename images, but it will generate a `rename.sh` file in the
output directory by default which will contain individual `mv` commands for renaming the
actual images. This gives a chance to review the filenames before actually renaming
everything, since AI is known to be somewhat unreliable at times.


Hints for best results
----------------------

This script will work better if your photos are organized in smaller directories, since
using too many people/dogs will require a larger context.
