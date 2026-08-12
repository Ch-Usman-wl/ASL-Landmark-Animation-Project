# Test folder scripts

This file lists the Python scripts in this folder and gives a short description of what each one does.

- **animate_book_skeleton.py**: Animate and visualize the skeletal/keypoint representation for "book" samples (skeleton-focused animation helper).
- **animate_clean.py**: Create animations from cleaned landmark data (renders cleaned/processed samples for inspection).
- **animate_sample.py**: Animate a single sample's landmarks or skeleton frames (general-purpose sample visualizer).
- **check_handedness.py**: Analyze samples to determine or verify left/right hand dominance and handedness consistency.
- **find_sample.py**: Search the dataset for samples matching given criteria (ID, label, or metadata) and return locations.
- **find_two_hand_sample.py**: Locate samples that contain two-handed gestures or landmarks (helper to find bimanual examples).
- **inspect_activity.py**: Inspect activity-level metadata across samples (summaries, counts, or activity tags).
- **inspect_book_hands.py**: Examine hand usage and annotations specifically for the "book" subset of samples.
- **inspect_frame.py**: Inspect and display a single frame from a sample (useful for debugging per-frame issues).
- **inspect_hands_clean.py**: Inspect cleaned/normalized hand landmark data and report any anomalies or stats.
- **inspect_hands.py**: Inspect raw hand landmarks in samples and provide visual or textual diagnostics.
- **inspect_motion.py**: Analyze motion characteristics across frames (velocity, acceleration, or motion summaries).
- **inspect_primarymath.py**: Tools to inspect the `primarymath` subset/dataset files and related metadata.
- **render_sample.py**: Render a sample to images or video (frame-by-frame rendering pipeline for visualization).
- **render_sequence.py**: Render a sequence of samples or a temporal sequence into a combined output (video/gif).
- **sample_label.py**: Utilities to inspect, generate, or modify sample labels (label parsing and display helpers).
- **tempCodeRunnerFile.py**: Temporary file created by the editor or runtime for quick ad-hoc experiments — not part of core tooling.
- **test_combine.py**: Test or example script that demonstrates combining multiple samples or data merges.
- **transitions.py**: Compute and visualize transitions between gestures/signs (transition detection and plotting utilities).
- **trim_sample.py**: Trim sample frames to a desired start/end range or remove irrelevant padding frames.
- **visualize_landmarks.py**: Visualization helpers for plotting landmark points and connections on frames.

Other files in this folder:

- **clean_1002.npy**: NumPy data file used by some inspection/render scripts (sample data blob).
- **BOOK_1027.gif**, **BOOK_1027.mp4**, **WAVE_1002.gif**, **WAVE_BOOK_transition.mp4**: Example media outputs produced by rendering scripts.

