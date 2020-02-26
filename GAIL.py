import torch.nn as nn

from PPO import PPO
from models.mlp_discriminator import Discriminator
from models.cnn_discriminator import CNNDiscriminator
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAIL(object):
    """
    A vanilla GAIL with JS-divergence Discriminator and PPO actor.
    """
    def __init__(self, args, state_dim, action_dim):
        self.device = device
        args.device = device
        self.config = args

        # initialize discriminator
        discriminator_action_dim = action_dim
        self.discriminator = Discriminator(state_dim+discriminator_action_dim).to(self.device)
        self.discriminator_loss = nn.BCEWithLogitsLoss()
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.config.learning_rate)
        # initialize actor
        self.policy = PPO(args, state_dim, action_dim)

    def expert_reward(self, state, action):
        """
        Compute the reward signal for the (PPO) actor update
        :param state:
        :param action:
        :return:
        """
        state_action = torch.FloatTensor(np.hstack([state, action]))
        with torch.no_grad():
            return -math.log(1-torch.sigmoid(self.discriminator(state_action)[0])+1e-8)

    def compute_entropy(self, logits):
        logsigmoid = nn.LogSigmoid()
        ent = (1.-torch.sigmoid(logits))*logits - logsigmoid(logits)
        return torch.mean(ent)

    def set_expert(self, expert_traj):
        """
        Set the expert trajectories.
        :param expert_traj:
        :return:
        """
        self.expert_traj = []
        # print(self.expert_traj[0])
        for traj in expert_traj:
            self.expert_traj.append(np.concatenate((traj[0], traj[2])))
        self.expert_traj = np.array(self.expert_traj)
        #     print(traj)
        # self.expert_state_actions
    def train_discriminator(self, batch):
        """
        Train the discriminator.
        :param batch:
        :return:
        """
        states = torch.FloatTensor(np.stack(batch['states'])).to(self.device)
        actions = torch.FloatTensor(np.stack(batch['actions'])).to(self.device)
        expert_state_actions = torch.FloatTensor(self.expert_traj).to(self.device)

        # assume one gradient step for now
        for _ in range(5):
            g_o = self.discriminator(torch.cat([states, actions], 1))
            e_o = self.discriminator(expert_state_actions)
            self.discriminator_optimizer.zero_grad()
            generator_loss = self.discriminator_loss(g_o, zeros((states.shape[0],1), device=self.device))
            expert_loss = self.discriminator_loss(e_o, ones((self.expert_traj.shape[0],1), device=self.device))

            # print(g_o.shape, e_o.shape)
            entropy_loss = self.compute_entropy(torch.cat([g_o, e_o], dim=0))
            # print(generator_loss, expert_loss, entropy_loss)
            # discrim_loss = generator_loss + expert_loss + 0.001 * entropy_loss
            discrim_loss = generator_loss + expert_loss + entropy_loss

            # compute accuracy
            with torch.no_grad():
                generator_accuracy = torch.mean((torch.sigmoid(g_o) < 0.5).float())
                expert_accuracy = torch.mean((torch.sigmoid(e_o) > 0.5).float())

            # discrim_loss = self.discriminator_loss(g_o, ones((states.shape[0], 1), device=self.device)) + \
            #     self.discriminator_loss(e_o, zeros((self.expert_traj.shape[0], 1), device=self.device))
            discrim_loss.backward()
            self.discriminator_optimizer.step()
        return {"d_loss": discrim_loss.to('cpu').detach().numpy(),
                "e_loss": expert_loss.to('cpu').detach().numpy(),
                "g_loss": generator_loss.to('cpu').detach().numpy(),
                "g_acc": generator_accuracy.to('cpu').detach().numpy(),
                "e_acc": expert_accuracy.to('cpu').detach().numpy()}

    def train(self, batch):
        """
        Train the discriminator and the actor.
        :param batch:
        :return:
        """

        loss = self.train_discriminator(batch)
        self.policy.train(batch)
        return loss

    def save(self, filename, directory):
        torch.save(self.policy.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
