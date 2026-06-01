



from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--config", type=str, required=True, help="Path to the config file to be used for testing.")

if __name__ == "__main__":
    args = args.parse_args()
    print(f"Received config path: {args.config}")
    