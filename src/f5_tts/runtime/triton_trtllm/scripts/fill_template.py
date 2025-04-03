#! /usr/bin/env python3
from argparse import ArgumentParser
from string import Template


def main(file_path, substitutions, in_place, participant_ids):
    with open(file_path) as f:
        pbtxt = Template(f.read())

    sub_dict = {"max_queue_size": 0}
    sub_dict["participant_ids"] = participant_ids
    for sub in substitutions.split(","):
        key, value = sub.split(":")
        sub_dict[key] = value

    pbtxt = pbtxt.safe_substitute(sub_dict)

    if in_place:
        with open(file_path, "w") as f:
            f.write(pbtxt)
    else:
        print(pbtxt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file_path", help="path of the .pbtxt to modify")
    parser.add_argument(
        "substitutions",
        help="substitutions to perform, in the format variable_name_1:value_1,variable_name_2:value_2...",
    )
    parser.add_argument("--in_place", "-i", action="store_true", help="do the operation in-place")
    parser.add_argument("--participant_ids", help="Participant IDs for the model", default="")
    args = parser.parse_args()

    main(**vars(args))
