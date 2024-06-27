import argparse


def parse_list_of_lists(value):
    try:
        # Split by ';' to separate different lists
        lists = value.split(";")
        # Further split each list by ',' to get individual elements
        return [lst.split(",") for lst in lists]
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid input for list of lists: {value}. Error: {e}"
        )
