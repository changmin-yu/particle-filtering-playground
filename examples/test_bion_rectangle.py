#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.
import matplotlib.pyplot as plt
import torch
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import time

from tqdm import trange
import pickle

from bion_rectangle.behav.slam_pixel_grid import (
    AgentState, 
    Control, 
    Retina, 
)
from bion_rectangle.environment import (
    GenerativeModelSingleEnvContTabular, 
    ObserverNav, 
)
from bion_rectangle.utils import numpytorch as np2
from bion_rectangle.behav.env_boundary import LargeSquare

torch.set_default_dtype(torch.float64)


def plot_particles(obs_cont, title=None):
    obs_cont.gen.env.plot_walls(mode='line')
    d = obs_cont.gen.env.height * 0.03
    plt.scatter(obs_cont.particles.loc[:, 0], obs_cont.particles.loc[:, 1], c='b', alpha=0.01)
    for i, (x, y, angle) in enumerate(obs_cont.particles.loc):
        angle = np.deg2rad(angle)
        diff = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]]) @ np.array([d, 0])
        plt.plot([x, x + diff[0]], [y, y + diff[1]], c='b', alpha=0.01)
        if i == torch.argmax(obs_cont.particles.weight):
            plt.scatter(x, y, c='k')
            plt.plot([x, x + diff[0]], [y, y + diff[1]], c='k')
    angle = np.deg2rad(obs_cont.gen.agent_state.heading_deg)
    diff = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]]) @ np.array([d, 0])
    for rot in [0, 90, 180, 270]:
        angle = np.deg2rad(rot)
        dif = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]]) @ diff
        pos = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]]) @ np.asarray(obs_cont.gen.agent_state.loc_xy)
        plt.plot([pos[0],
                  pos[0] + dif[0]],
                 [pos[1], pos[1] + dif[1]], c='r')
        plt.scatter(pos[0], pos[1], c='r')

    if title is not None:
        plt.title(f"Iteration {title}")
    plt.show()
    return


def plot_visual(obs_cont, loc=None, i=None):
    if loc is not None:
        state = AgentState()
        state.loc_xy, state.heading_deg = torch.tensor((loc[0], loc[1])), torch.tensor(loc[2])
    else:
        state = obs_cont.gen.agent_state
    plt.imshow(np2.npy(obs_cont.gen.measure_retinal_image(state)).transpose((1, 0, 2)),
               origin='lower'
               )
    if i is not None:
        if loc is None:
            plt.title(f"Agent: Iteration {i}, {state.loc_xy}, {state.heading_deg}")
        else:
            plt.title(f"Max: Iteration {i}, {state.loc_xy}, {state.heading_deg}")
    else:
        plt.title(f"{state.loc_xy}, {state.heading_deg}")
    plt.show()
    return


def update_state(obs_cont, control):
    dh = control.dheading_deg[-1] * obs_cont.dt
    angle = np.deg2rad(obs_cont.gen.agent_state.heading_deg + dh)
    dxdy = (np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]) @
            np.array([control.velocity_ego[0], control.velocity_ego[1]])) * obs_cont.dt
    obs_cont.gen.agent_state.set_state((obs_cont.gen.agent_state.loc_xy[0] + dxdy[0],
                                    obs_cont.gen.agent_state.loc_xy[1] + dxdy[1]),
                                   obs_cont.gen.agent_state.heading_deg + dh)
    return


def main():
    t0 = time.time()
    env = LargeSquare(contrast=0.5, contrast_btw_walls=0.0, height_wall=.3) # contrast between walls, consider 0.3
    retina = Retina(fov_deg=(90., 90.), deg_per_pix=2.)
    gen_cont = GenerativeModelSingleEnvContTabular(env=env, retina=retina)
    obs_cont = ObserverNav(gen_cont, (0., 0.),
                                .85, (0.3, 0.))


    agent_pos = torch.tensor([0.3, 0.3, 120.])
    control = Control(15, (0.1, 0.))
    obs_cont.gen.agent_state.set_state((agent_pos[0], agent_pos[1]), agent_pos[2])
    obs_cont.measurement_step(obs_cont.gen.meas, update_state=True)

    # plot_particles(obs_cont)
    # plot_visual(obs_cont)
    # plot_visual(obs_cont, loc=obs_cont.particles.loc[torch.argmax(obs_cont.particles.weight)])
    obs_cont.resample_cont(MH=False)
    obs_cont.measurement_step(obs_cont.gen.meas, update_state=True)

    # plot_visual(obs_cont, i=-1)
    # plot_particles(obs_cont)
    obs_cont.particles.downsample(2048)
    # plot_particles(obs_cont)
    
    print(f"Initial processing time: {time.time() - t0:.2f}")

    for i in range(10):
        print(i)
        update_state(obs_cont, control)

        obs_cont.transition_step(control=control)
        obs_cont.measurement_step(obs_cont.gen.meas, update_state=True)
        # plot_particles(obs_cont, title=(i, 'predicted', 1 / torch.sum(obs_cont.particles.weight ** 2)))

        if (1 / torch.sum(obs_cont.particles.weight ** 2) <
            0.1 * obs_cont.particles.n_particle) and i % 5 == 4:
            obs_cont.resample_cont(MH=False)
            # plot_particles(obs_cont, title=(i, 'resampled'))


def main_for_evaluation(num_seeds: int = 100, num_timestep: int = 20, num_particles: int = 5000):
    particle_locs_and_hd = np.zeros((num_seeds, num_timestep+1, num_particles, 3))
    particle_weights = np.zeros((num_seeds, num_timestep+1, num_particles))
    true_locs = np.zeros((num_seeds, num_timestep+1, 2))
    true_hd = np.zeros((num_seeds, num_timestep+1, 1))
    
    with trange(num_seeds, dynamic_ncols=True) as pbar:
        for i in pbar:
            t0 = time.time()
            env = LargeSquare(contrast=0.5, contrast_btw_walls=0.0, height_wall=.3) # contrast between walls, consider 0.3
            retina = Retina(fov_deg=(90., 90.), deg_per_pix=2.)
            gen_cont = GenerativeModelSingleEnvContTabular(env=env, retina=retina)
            obs_cont = ObserverNav(gen_cont, (0., 0.),
                                        .85, (0.3, 0.), 
                                        init_num_particles=num_particles)


            agent_pos = torch.tensor([0.3, 0.3, 120.])
            control = Control(15, (0.1, 0.))
            obs_cont.gen.agent_state.set_state((agent_pos[0], agent_pos[1]), agent_pos[2])
            obs_cont.measurement_step(obs_cont.gen.meas, update_state=True)

            obs_cont.resample_cont(MH=False)
            obs_cont.measurement_step(obs_cont.gen.meas, update_state=True)
            
            particle_locs_and_hd[i][0] = obs_cont.particles.loc
            particle_weights[i][0] = obs_cont.particles.weight
            true_locs[i][0] = obs_cont.gen.agent_state.loc_xy
            true_hd[i][0] = obs_cont.gen.agent_state.heading_deg

            pbar.set_description(f"Initialisation complete, time taken: {time.time()-t0:.2f}s")
            
            for j in range(num_timestep):
                t0 = time.time()
                update_state(obs_cont, control)

                obs_cont.transition_step(control=control)
                obs_cont.measurement_step(obs_cont.gen.meas, update_state=True)
                # plot_particles(obs_cont, title=(i, 'predicted', 1 / torch.sum(obs_cont.particles.weight ** 2)))

                if (1 / torch.sum(obs_cont.particles.weight ** 2) < 0.5 * obs_cont.particles.n_particle): #  and i % 5 == 4:
                    obs_cont.resample_cont(MH=False)
                    
                particle_locs_and_hd[i][j+1] = obs_cont.particles.loc
                particle_weights[i][j+1] = obs_cont.particles.weight
                true_locs[i][j+1] = obs_cont.gen.agent_state.loc_xy
                true_hd[i][j+1] = obs_cont.gen.agent_state.heading_deg
                
                pbar.set_description(f"Seed {i}, ts {j}, time taken: {time.time()-t0:.2f}s")
    
    with open(f"logdir/test_bion_rectangle/S{num_seeds}_T{num_timestep}_P{num_particles}.pkl", "wb") as f:
        pickle.dump(
            {
                "particle_state": particle_locs_and_hd, 
                "particle_weight": particle_weights, 
                "true_locs": true_locs, 
                "true_hd": true_hd, 
            }, 
            f
        )
    f.close()
    
    return {
        "particle_state": particle_locs_and_hd, 
        "particle_weight": particle_weights, 
        "true_locs": true_locs, 
        "true_hd": true_hd, 
    }


if __name__ == '__main__':
    # main()  
    num_seeds = 20
    num_timestep = 20
    num_particles = 2048
    main_for_evaluation(num_seeds, num_timestep, num_particles)
