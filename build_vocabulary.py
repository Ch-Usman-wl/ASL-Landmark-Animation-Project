# # import h5py
# # import json

# # HDF5_PATH = "data/primarymath/val.hdf5"
# # OUTPUT_PATH = "word_samples.json"


# # def decode_value(value):
# #     """
# #     Convert an HDF5 scalar/string value into a normal Python string.
# #     """

# #     if hasattr(value, "shape") and value.shape == ():
# #         value = value[()]

# #     if isinstance(value, bytes):
# #         return value.decode("utf-8")

# #     return str(value)


# # def build_vocabulary():
# #     vocabulary = {}

# #     with h5py.File(HDF5_PATH, "r") as f:

# #         for sample_id in f.keys():

# #             group = f[sample_id]

# #             if "label" not in group:
# #                 continue

# #             label = decode_value(
# #                 group["label"]
# #             ).strip()

# #             if not label:
# #                 continue

# #             sample_id = str(sample_id)

# #             # Keep the first sample we encounter.
# #             if label not in vocabulary:
# #                 vocabulary[label] = sample_id

# #     return vocabulary


# # if __name__ == "__main__":

# #     vocabulary = build_vocabulary()

# #     with open(
# #         OUTPUT_PATH,
# #         "w",
# #         encoding="utf-8"
# #     ) as f:

# #         json.dump(
# #             vocabulary,
# #             f,
# #             indent=4,
# #             ensure_ascii=False
# #         )

# #     print(
# #         f"Found {len(vocabulary)} unique words."
# #     )

# #     print()
# #     print("Vocabulary:")

# #     for word, sample_id in sorted(
# #         vocabulary.items()
# #     ):
# #         print(
# #             f"{word}: {sample_id}"
# #         )

# #     print()
# #     print(
# #         f"Saved to: {OUTPUT_PATH}"
# #     )


# import h5py
# import json

# HDF5_PATH = "data/primarymath/val.hdf5"
# LABELS_PATH = "data/primarymath/labels.json"
# OUTPUT_PATH = "word_samples.json"


# def decode_value(value):
#     """Convert an HDF5 scalar/string value into a Python string."""

#     if hasattr(value, "shape") and value.shape == ():
#         value = value[()]

#     if isinstance(value, bytes):
#         return value.decode("utf-8")

#     return str(value)


# def load_label_mapping():
#     """Load numeric class ID -> word mapping."""

#     with open(
#         LABELS_PATH,
#         "r",
#         encoding="utf-8"
#     ) as f:

#         labels = json.load(f)

#     return labels["id_to_label"]


# def build_vocabulary():

#     id_to_label = load_label_mapping()

#     vocabulary = {}

#     with h5py.File(
#         HDF5_PATH,
#         "r"
#     ) as f:

#         for sample_id in f.keys():

#             group = f[sample_id]

#             if "label" not in group:
#                 continue

#             label_id = decode_value(
#                 group["label"]
#             ).strip()

#             if label_id not in id_to_label:
#                 continue

#             word = id_to_label[
#                 label_id
#             ]

#             sample_id = str(sample_id)

#             # Keep the first sample for each word.
#             if word not in vocabulary:

#                 vocabulary[word] = sample_id

#     return vocabulary


# if __name__ == "__main__":

#     vocabulary = build_vocabulary()

#     with open(
#         OUTPUT_PATH,
#         "w",
#         encoding="utf-8"
#     ) as f:

#         json.dump(
#             vocabulary,
#             f,
#             indent=4,
#             ensure_ascii=False
#         )

#     print(
#         f"Found {len(vocabulary)} unique words."
#     )

#     print()
#     print("Vocabulary:")

#     for word, sample_id in sorted(
#         vocabulary.items()
#     ):

#         print(
#             f"{word}: {sample_id}"
#         )

#     print()
#     print(
#         f"Saved to: {OUTPUT_PATH}"
#     )



import h5py
import json

HDF5_PATH = "data/primarymath/val.hdf5"
LABELS_PATH = "data/primarymath/labels.json"
OUTPUT_PATH = "word_samples.json"


def decode_value(value):
    """Convert an HDF5 scalar/string value into a Python string."""

    if hasattr(value, "shape") and value.shape == ():
        value = value[()]

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


def load_label_mapping():
    """Load numeric class ID -> word mapping."""

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        labels = json.load(f)

    return labels["id_to_label"]


def build_vocabulary():

    id_to_label = load_label_mapping()

    vocabulary = {}

    with h5py.File(
        HDF5_PATH,
        "r"
    ) as f:

        for sample_id in f.keys():

            group = f[sample_id]

            if "label" not in group:
                continue

            label_id = decode_value(
                group["label"]
            ).strip()

            if label_id not in id_to_label:
                continue

            word = id_to_label[label_id]

            sample_id = str(sample_id)

            if word not in vocabulary:
                vocabulary[word] = []

            vocabulary[word].append(
                sample_id
            )

    return vocabulary


if __name__ == "__main__":

    vocabulary = build_vocabulary()

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            vocabulary,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Found {len(vocabulary)} unique words."
    )

    print()

    for word, samples in sorted(
        vocabulary.items()
    ):

        print(
            f"{word}: {len(samples)} samples"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )