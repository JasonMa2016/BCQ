import numpy as np
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_trajs", default=5, type=int)
    parser.add_argument("--model", default="DRBCQ")
    args = parser.parse_args()

    results = np.load("./results/" + "{}_traj{}_{}_{}.npy".format(args.model, args.num_trajs,
                                                                  args.env_name, args.seed))
    print(np.mean(results, axis=1))
    # print(np.std(results,axis=1))