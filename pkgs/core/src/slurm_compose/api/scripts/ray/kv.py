import argparse
import json
import logging
import sys

import ray


@ray.remote
class KVStore:
    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value
        return value

    def get(self, key):
        return self.store.get(key, None)

    def getall(self):
        return self.store

    def delete(self, key):
        return self.store.pop(key, None)


def main():
    parser = argparse.ArgumentParser(description="Ray In-Memory KV Store CLI")
    parser.add_argument("--address", type=str, default="localhost:10001", help="Ray client address.")
    parser.add_argument("--namespace", type=str, default="slurm", help="Ray namespace.")
    parser.add_argument("--name", type=str, default="global_kv", help="Ray actor name.")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("init", help="Initialize the detached KV store actor.")

    get_parser = subparsers.add_parser("get", help="Get a value.")
    get_parser.add_argument("--key", type=str)

    subparsers.add_parser("getall", help="Get the complete state of current store.")

    set_parser = subparsers.add_parser("set", help="Set a value.")
    set_parser.add_argument("--key", type=str)
    set_parser.add_argument("--value", type=str)

    args = parser.parse_args()

    try:
        ray.init(address=f"ray://{args.address}", namespace=args.namespace, logging_level=logging.ERROR)

        if args.command == "init":
            KVStore.options(name=args.name, lifetime="detached", get_if_exists=True).remote()

            print(f"[INFO] KV actor {args.namespace}/{args.name} is live.", file=sys.stderr)
        elif args.command == "get":
            kv = ray.get_actor(args.name)

            value = ray.get(kv.get.remote(args.key))
            if value is None:
                sys.exit(1)

            print(value)
        elif args.command == "getall":
            kv = ray.get_actor(args.name)

            print(json.dumps(ray.get(kv.getall.remote())))
        elif args.command == "set":
            kv = ray.get_actor(args.name)
            ray.get(kv.set.remote(args.key, args.value))

            print(
                f"[INFO] Set key {args.key} with value {args.value} in {args.namespace}/{args.name}.", file=sys.stderr
            )
        else:
            raise NotImplementedError
    except (ray.exceptions.RayError, ValueError, ConnectionError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
