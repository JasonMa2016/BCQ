import numpy as np

"""Wrapper for a discrete Markov decision process that makes shape checks"""
class MDP(object):
    def __init__(self, T, R, discount):
        """Initialize the Markov Decision Process.
        - `T` should be a 3D array whose dimensions represent initial states,
          actions, and next states, respectively, and whose values represent
          transition probabilities.
        - `R` should be a 1D array describing rewards for beginning each
          timestep in a particular state (or a 3D array like `T`). It will be
          transformed into the appropriate 3D shape.
        - `discount` should be a value in [0,1) controlling the decay of future
          rewards."""
        Ds, Da, _ = T.shape
        if T.shape not in [(Ds, Da, Ds)]:
            raise ValueError("T should be in R^|S|x|A|x|S|")
        if R.shape not in [(Ds, Da, Ds), (Ds,)]:
            raise ValueError("R should be in R^|S| or like T")
        if discount < 0 or discount >= 1:
            raise ValueError("discount should be in [0,1)")
        if R.shape == (Ds,): # Expand R if necessary
            R = np.array([[[R[s1] for s2 in range(Ds)]
                for a in range(Da)]
                for s1 in range(Ds)])
        self.T = T
        self.R = R
        self.discount = discount
        self.num_states = Ds
        self.num_actions = Da
        self.states = np.arange(Ds)
        self.actions = np.arange(Da)

def iterative_value_estimation(mdp, policy, tol=1e-5):
    """Value estimation algorithm from page 75, Sutton and Barto. Returns an
    estimate of the value of a given policy under the MDP (with the number of
    iterations required to reach specified tolerance)."""
    V = np.zeros(mdp.num_states)
    num_iters = 0

    # Compute transition and reward matrices specific to this policy
    Tp = np.array([mdp.T[i,a] for i,a in enumerate(policy)])
    Rp = np.array([mdp.R[i,a] for i,a in enumerate(policy)])
    rewards_by_state = (Tp * Rp).sum(axis=1) # Exp. rewards for each state

    while True:
        # Apply updates state-by-state
        V_new = np.array(V)
        for s, state_reward in enumerate(rewards_by_state):
            next_val = np.dot(Tp[s], V_new)
            V_new[s] = state_reward + mdp.discount * next_val
        # Check if converged
        if np.abs(V - V_new).max() < tol: break
        # Continue iterating if not
        V = V_new
        num_iters += 1

    return V, num_iters

def Q_function(mdp, policy, tol=1e-5):
    """Q function from Equation 4.6, Sutton and Barto. For each state and
    action, returns the value of performing the action at that state, then
    following the policy thereafter."""
    V, _ = iterative_value_estimation(mdp, policy, tol=tol)
    state_rewards = (mdp.T * mdp.R).sum(axis=2)
    next_s_values = np.dot(mdp.T, V)
    Q = state_rewards + mdp.discount * next_s_values
    assert(Q.shape == (mdp.num_states, mdp.num_actions))
    return Q

def policy_iteration(mdp, init_policy=None, tol=1e-5):
    """Policy iteration algorithm from page 80, Sutton and Barto.
    Iteratively transform the initial policy to become optimal.
    Return the full path."""
    if init_policy is None:
        init_policy = np.zeros(mdp.num_states, dtype=int)
    policies = [np.array(init_policy)]

    while True:
        old_policy = policies[-1]
        new_policy = np.argmax(Q_function(mdp, old_policy, tol=tol), axis=1)
        if np.array_equal(old_policy, new_policy): break
        policies.append(new_policy)

    return policies

def build_homework0_mdp():
    """Build an MDP representing the setting described in homework0.pdf."""
    states = np.array([0,1,2,3,4])
    actions = np.array([0,1])
    def next_state_probs(s, a):
        transition = np.zeros_like(states)
        next_state = max(s-1, 0) if a == 0 else min(s+1, 4)
        transition[next_state] = 1.0
        return transition
    T = np.array([[next_state_probs(s, a) for a in actions] for s in states])
    R = np.array([-1,-1,-1,-1,10])
    return MDP(T, R, 0.75)

def build_always_left_policy():
    """Build a policy representing the action "left" in every state."""
    return np.array([0,0,0,0,0])
