from fractalai.model import RandomDiscreteModel
from fractalai.environment import ExternalProcess, ParallelEnvironment, AtariEnvironment
from fractalai.fractalmc import FractalMC



name = "MontezumaRevenge-ram-v0"
render = False # It is funnier if the game is displayed on the screen
clone_seeds = True  # This will speed things up a bit
max_steps = 1e6  # Play until the game is finished.
n_repeat_action = 1  # Atari games run at 20 fps, so taking 4 actions per seconds is more
reward_limit = 20000
render_every = 2
dt_mean = 3
dt_std = 2
min_dt = 3


max_samples = 6000  # Let see how well it can perform using at most 300 samples per step
max_walkers = 100 # Let's set a really small number to make everthing faster
time_horizon = 30  # 50 frames should be enough to realise you have been eaten by a ghost

env = ParallelEnvironment(name=name,env_class=AtariEnvironment,
                          blocking=False, n_workers=16, n_repeat_action=n_repeat_action)  # We will play an Atari game
model = RandomDiscreteModel(max_wakers=max_walkers,
                            n_actions=env.n_actions) # The Agent will take discrete actions at random

fmc = FractalMC(model=model, env=env, max_walkers=max_walkers,
                reward_limit=reward_limit, render_every=render_every,
                time_horizon=time_horizon, dt_mean=dt_mean, dt_std=dt_std, accumulate_rewards=True, min_dt=min_dt)


fmc.run_agent(render=False, print_swarm=False)
#fmc.render_game(sleep=1/40)