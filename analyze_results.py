import numpy as np
import argparse
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_trajs", default=5, type=int)
    parser.add_argument("--model", default="DRBCQ")
    args = parser.parse_args()

    models = ['BCQ', 'DRBCQ', 'BC']
    types = ['good', 'mixed']

    fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

    # for i, type in enumerate(types):
    #     for model in models:
    #         if model == 'BC':
    #             file_name = "./results/" + "{}_{}_traj{}_seed{}_sample0_{}.npy".format(model, args.env_name, args.num_trajs, args.seed,
    #                                                                    type)
    #             model_performance = np.load(file_name)
    #             axs[i].plot([10*i for i in range(10)], np.mean(model_performance, axis=1), label=model)
    #             axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
    #             axs[i].legend(loc='upper right')
    #         else:
    #             file_name = "./results/" + "{}_{}_traj{}_seed{}_{}.npy".format(model, args.env_name, args.num_trajs, args.seed,
    #                                                                    type)
    #             model_performance = np.load(file_name)
    #             axs[i].plot([i for i in range(100)],np.mean(model_performance, axis=1), label=model)
    #             axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
    #             axs[i].legend(loc='upper right')

    for i, model in enumerate(models):
        for type in types:
            if model == 'BC':
                file_name = "./results/" + "{}_{}_traj{}_seed{}_sample5_{}.npy".format(model, args.env_name, args.num_trajs, args.seed,
                                                                       type)
            else:
                file_name = "./results/" + "{}_{}_traj{}_seed{}_{}.npy".format(model, args.env_name, args.num_trajs, args.seed,
                                                                               type)
            model_performance = np.load(file_name)
            axs[i].plot([i for i in range(model_performance.shape[0])],np.mean(model_performance, axis=1), label=type)
            axs[i].set_title('{} {} Training Curve'.format(type, args.env_name))
            axs[i].legend(loc='upper right')
    plt.show()
    # results = np.load("./results/" + "{}_traj{}_{}_{}.npy".format(args.model, args.num_trajs,
    #                                                               args.env_name, args.seed))
    # print(np.mean(results, axis=1))
    # print(np.shape(results))
    #
    # results = np.load("./results/" + "{}_traj{}_{}_{}_good.npy".format(args.model, args.num_trajs,
    #                                                               args.env_name, args.seed))
    # print(np.mean(results, axis=1))
    # print(np.shape(results))
    #
    #
    # results = np.load("./results/" + "{}_traj{}_{}_{}_mixed.npy".format(args.model, args.num_trajs,
    #                                                               args.env_name, args.seed))
    # print(np.mean(results, axis=1))
    # print(np.shape(results))
    #
    #

    # print(np.std(results,axis=1))