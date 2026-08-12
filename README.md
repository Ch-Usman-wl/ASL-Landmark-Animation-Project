# ASL Landmark Animation Project

A Python project for generating **American Sign Language (ASL) landmark animations** from text.

The project combines static ASL fingerspelling landmarks with temporal ASL sign data. The goal is to convert text into a sequence of hand landmarks that can be rendered as an animated skeleton.

## Overview

The project currently has two main sources of ASL data:

1. **ASLNow Fingerspelling**

   * Static hand poses for alphabet letters.
   * MediaPipe hand landmarks.
   * Used for fingerspelling words that are not available as complete signs.
   * Currently supports the available **A–W** letters from the downloaded dataset.

2. **MS-ASL / PrimaryMath temporal data**

   * Video-derived landmark sequences.
   * Used for complete ASL signs such as `BOOK`, `SCHOOL`, and other words available in the dataset.
   * These sequences contain actual movement over time.

The project can therefore work toward a system such as:

```text
Input text
    |
    +-- Known ASL sign
    |       |
    |       v
    |   Temporal landmarks
    |
    +-- Unknown word
            |
            v
       Fingerspelling
            |
            v
      Letter landmarks
            |
            v
       Final sequence
            |
            v
        Renderer
            |
            v
       Animated GIF
```

## Current Scope

The project intentionally focuses on the data that is currently reliable and usable.

### Static fingerspelling

Currently available from ASLNow:

```text
A B C D E F G H I K L M N O P Q R S T U V W
```

The project does **not** currently attempt to synthesize:

```text
J X Y Z
```

J and Z are motion-based ASL letters, while the available datasets investigated for X/Y did not provide a sufficiently compatible raw-landmark representation for this project.

Rather than introducing unreliable or incompatible data, these letters are currently left unsupported.

If a future temporal ASL dataset/model provides appropriate landmark sequences for J or Z, they can be added later.

## Landmark Representation

The underlying ASLNow JSON files contain MediaPipe hand landmarks.

Each hand contains:

```text
21 landmarks
```

and each landmark contains:

```text
x
y
z
```

The current animation pipeline uses the 2D:

```text
x
y
```

coordinates.

The project internally uses a 135-landmark representation inherited from the temporal dataset.

The hand regions used by the renderer are:

```text
91:112   Hand A
112:133  Hand B
```

For fingerspelling, the selected hand is placed consistently into the hand region used by the current pipeline.

## Project Structure

The important project files are:

```text
proj_1/
│
├── data/
│   ├── aslnow/
│   │   ├── A/
│   │   ├── B/
│   │   ├── C/
│   │   └── ...
│   │
│   └── primarymath/
│
├── asl_data.py
├── composer.py
├── transitions.py
├── renderer.py
├── test_fingerspell.py
│
└── extra_letters_test/
```

### `asl_data.py`

Loads landmark sequences from the temporal ASL dataset.

The loaded temporal data is converted into the representation used by the rest of the project.

### `transitions.py`

Creates interpolated landmark transitions between two sequences.

The current fingerspelling configuration uses:

```text
8 transition frames
```

### `composer.py`

Provides utilities for:

* trimming empty landmark frames
* composing multiple temporal sequences
* inserting transitions between sequences

### `renderer.py`

Renders a `(frames, 135, 2)` landmark sequence as a 2D animated hand skeleton.

The renderer:

* extracts the two hand regions
* draws the 21 hand landmarks
* draws the standard hand connections
* animates the sequence
* can save the result as a GIF

### `test_fingerspell.py`

Generates static fingerspelling animations from ASLNow.

It handles:

* selecting compatible hand orientation
* selecting representative samples
* normalizing hand size/position
* converting a static hand into the 135-landmark representation
* holding each letter for multiple frames
* transitions between letters
* repeated-letter handling
* special scaling for certain letters

## Fingerspelling Pipeline

A fingerspelled word is processed approximately as follows:

```text
WORD
 |
 v
Split into letters
 |
 v
Find ASLNow sample for each letter
 |
 v
Check hand orientation
 |
 v
Normalize hand
 |
 v
Convert to 135-landmark representation
 |
 v
Apply letter-specific scaling
 |
 v
Hold pose
 |
 v
Insert transitions
 |
 v
Handle repeated letters
 |
 v
Complete landmark sequence
 |
 v
Render GIF
```

## Representative Samples

ASLNow contains multiple samples for each letter.

Rather than arbitrarily selecting a sample, the project can filter samples by hand orientation and select a representative pose.

Some letters also have manually selected samples that were visually inspected.

These selections can be updated as better samples are discovered.

## Hand Orientation

The ASLNow dataset contains examples with different hand orientations.

The fingerspelling pipeline currently selects a consistent orientation so that a word does not alternate between different hands or mirrored-looking poses.

This is particularly important when composing multiple letters:

```text
B → O → O → K
```

The goal is for all four letters to appear to originate from the same hand rather than having the hand suddenly morph from one side/orientation to another.

## Letter Scaling

Some static ASLNow poses appeared disproportionately large or extended when combined with other letters.

The current pipeline applies a small reduction to:

```text
M
N
E
O
```

The current scale factor is:

```text
0.75
```

The wrist remains fixed while the fingers are scaled toward it.

This is a practical normalization adjustment rather than a modification to the underlying ASL data.

## Repeated Letters

Repeated letters need special handling.

For example:

```text
BOOK
```

contains:

```text
B O O K
```

Simply holding the O pose twice can make the result appear to be one long O.

The current solution inserts a small release/opening movement between repeated letters.

Current settings:

```text
FRAMES_PER_LETTER = 12
TRANSITION_FRAMES = 8
REPEAT_RELEASE_FRAMES = 5
REPEAT_OPEN_AMOUNT = 0.10
```

The release is intentionally small so that:

```text
O → O
```

is visually distinguishable without creating an unrealistic large movement.

## Rendering

The renderer produces a 2D skeleton from the landmark coordinates.

The hand uses the standard 21-landmark structure:

```text
0  = wrist

1–4   = thumb
5–8   = index
9–12  = middle
13–16 = ring
17–20 = pinky
```

The palm connections are also drawn between:

```text
5 → 9
9 → 13
13 → 17
```

The renderer can save the animation as a GIF using Pillow.

## Example

A fingerspelled word can be generated by setting the word in the fingerspelling script:

```python
WORD = "BOOK"
```

The resulting landmark sequence is then passed to the existing renderer.

The expected workflow is:

```text
BOOK
 ↓
B
 ↓
O
 ↓
O
 ↓
K
 ↓
combined landmark sequence
 ↓
GIF
```

## Temporal ASL Signs

The project also has temporal landmark data from the MS-ASL/PrimaryMath dataset.

Unlike static fingerspelling:

```text
B → O → O → K
```

a temporal sign such as `BOOK` is represented by a sequence of changing landmarks:

```text
frame 1
frame 2
frame 3
...
frame N
```

This allows the project to use actual sign movement when a word is available in the temporal dataset.

## Combining Signs and Fingerspelling

The intended final architecture is:

```text
Input sentence
      |
      v
Split into words
      |
      v
Check each word
      |
      +----------------------+
      |                      |
      v                      v
Known temporal sign       Not available
      |                      |
      v                      v
Temporal sequence         Fingerspell
      |                      |
      +----------+-----------+
                 |
                 v
       Word-level transition
                 |
                 v
        Complete sequence
                 |
                 v
             Renderer
```

For example:

```text
BOOK SCHOOL
```

could eventually become:

```text
BOOK sign
    ↓
transition
    ↓
SCHOOL sign
```

while an unsupported word could fall back to:

```text
B → O → O → K
```

using the fingerspelling pipeline.

This combined word/sign system is the next major stage of the project.

## Data Quality and Limitations

The datasets used in this project are not perfectly standardized.

Different sources may differ in:

* landmark ordering
* coordinate systems
* hand orientation
* number of landmarks
* temporal resolution
* static vs. temporal representation
* missing landmarks
* scale and positioning

For this reason, the project deliberately does **not** attempt to force every available ASL dataset into one representation.

The current goal is to use reliable subsets of the available data and keep the conversion steps explicit.

## Current Limitations

The project currently has several limitations:

* Static fingerspelling is limited to the available ASLNow alphabet.
* J and Z are not currently supported as dynamic fingerspelling gestures.
* X and Y are not currently included.
* Static poses do not represent the full natural motion of a human signer.
* The landmark renderer is currently 2D.
* Hand appearance and scale can vary between source samples.
* Some manually selected poses may require further refinement.
* The temporal sign dataset contains only its available vocabulary.
* Sentence-level ASL grammar is not yet implemented.

The system should therefore be considered a **landmark animation/composition system**, not a complete ASL translation system.

## Future Work

Potential future improvements include:

### 1. Sign + fingerspelling integration

Automatically choose between:

```text
known temporal sign
```

and:

```text
fingerspelling fallback
```

for each word.

### 2. Sentence-level composition

Allow complete text input:

```text
"HELLO MY FRIEND"
```

and generate one continuous landmark sequence.

### 3. Better transitions

Improve transitions between:

* letters
* repeated letters
* complete signs
* signs and fingerspelling

### 4. 3D rendering

The original landmark data contains Z coordinates, so a future renderer could use:

```text
x
y
z
```

instead of only:

```text
x
y
```

### 5. Learned landmark generation

A future model could learn to generate temporal landmark sequences directly rather than relying entirely on manually selected static poses and interpolation.

### 6. Improved ASL vocabulary

Additional temporal signs can be added as the dataset/model vocabulary grows.

## Philosophy

This project prioritizes **usable and consistent landmark data over artificially filling gaps**.

When a dataset does not provide reliable information for a letter or gesture, the project leaves that gesture unsupported instead of fabricating a representation that looks plausible but does not correspond to the actual ASL movement.

The long-term goal is to combine:

```text
Reliable ASL data
        +
Temporal landmark modeling
        +
Fingerspelling
        +
Landmark generation
        +
Rendering
```

into a single system capable of producing coherent ASL landmark animations from text.
