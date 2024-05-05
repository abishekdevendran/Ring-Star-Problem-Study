import os


def tspLibrary(dir: str):
    """Reads the tsp files in the directory and returns a list[list[floats]]"""
    with open(dir) as f:
        lines = f.readlines()
    data = [x.strip().split() for x in lines]
    # cleanup the data where count of elements is not equal to 3
    data = [x for x in data if len(x) == 3]
    # skip the lines where the first element is not a number
    while not data[0][0].strip().isnumeric():
        data = data[1:]
    data = [[float(x) for x in y] for y in data]
    # drop the first element of all the lists
    data = [x[1:] for x in data]
    return data


def loadAllTSPs(directory: str):
    """Loads all tsp files in the directory"""
    files = os.listdir(directory)
    files = [x for x in files if x.endswith('.tsp')]
    data = []
    for file in files:
        try:
            data.append(tspLibrary(directory + file))
        except Exception as e:
            # print(f"Error in reading file {file}")
            # print(e)
            pass
    return data
