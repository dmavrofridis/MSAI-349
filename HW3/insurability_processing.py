def seperate_insurability(input):
    labels = []
    attributes = []

    for i in input:
        labels.append(i[0][0])
        attributes.append(i[1])
    return labels, attributes


def read_insurability(file_name):

    count = 0
    data = []
    with open(file_name, "rt") as f:
        for line in f:
            if count > 0:
                line = line.replace("\n", "")
                tokens = line.split(",")
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == "Good":
                        cls = 0
                    elif tokens[3] == "Neutral":
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls], [x1, x2, x3]])
            count = count + 1
    return data