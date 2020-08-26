import json


def main():

    path = "./_context.json"

    try:
        with open(path, 'r') as fin:
            context = json.load(fin)
    except Exception as e:
        raise Exception("failed to parse context file %s: %s" % (path, e))

    print(context["s3uri"])


if __name__ == "__main__":
    main()
